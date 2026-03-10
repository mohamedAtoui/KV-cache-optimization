"""Two-stage calibration for KV2State: decay alignment + LoRA fine-tuning.

Stage 1: Per-head MSE alignment — learns optimal decay (log_decay) per streaming head
         by matching softmax attention outputs as teacher. All other params frozen.

Stage 2: End-to-end LoRA fine-tuning — adds LoRA adapters to q_proj/v_proj,
         freezes calibrated decay, trains with cross-entropy LM loss.

Reference: LoLCATs (Hedgehog + LoRA), Mamba-2 distillation literature.
"""

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kv2state.hybrid_attention import _apply_rotary_pos_emb, _feature_map

logger = logging.getLogger(__name__)


def _get_streaming_data_iterator(
    tokenizer,
    dataset_name: str = "DKYoon/SlimPajama-6B",
    dataset_subset: str = "train",
    seq_len: int = 512,
    batch_size: int = 4,
):
    """Create a streaming data iterator from SlimPajama (or compatible).

    Yields batches of tokenized input_ids without loading the full dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=dataset_subset, streaming=True)

    buffer = []
    for example in ds:
        text = example.get("text", "")
        if len(text.strip()) < 50:
            continue
        tokens = tokenizer(text, truncation=True, max_length=seq_len, return_tensors="pt")
        ids = tokens["input_ids"].squeeze(0)
        if ids.shape[0] < seq_len:
            continue
        buffer.append(ids)
        if len(buffer) == batch_size:
            yield torch.stack(buffer)
            buffer = []


def calibrate_stage1(
    model: nn.Module,
    tokenizer,
    head_classification,
    state_cache,
    dataset_name: str = "DKYoon/SlimPajama-6B",
    dataset_subset: str = "train",
    lr: float = 0.02,
    num_steps: int = 500,
    batch_size: int = 2,
    seq_len: int = 128,
    warmup_steps: int = 50,
    log_interval: int = 50,
    device: str = "cuda",
) -> dict:
    """Stage 1: Per-head MSE alignment to learn optimal decay per streaming head.

    For each batch:
    1. Run original model (teacher) to get per-layer hidden states
    2. For each layer with streaming heads, compute teacher attention output
       using standard softmax, and student output using the patched state attention
    3. MSE loss between teacher and student on streaming head outputs only
    4. Only log_decay parameters are trainable

    Args:
        model: Patched KV2State model (with _kv2state_modules).
        tokenizer: Tokenizer for the model.
        head_classification: HeadClassification from head_classifier.
        state_cache: StateCache (will be reset each batch).
        dataset_name: HuggingFace dataset for calibration data.
        dataset_subset: Dataset split.
        lr: Learning rate for decay parameters.
        num_steps: Number of training steps.
        batch_size: Batch size.
        seq_len: Sequence length per sample.
        warmup_steps: Linear warmup steps.
        log_interval: Steps between logging.
        device: Device to train on.

    Returns:
        Dict with training stats: losses, final decay values, etc.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Per-head MSE alignment (learning decay)")
    logger.info("=" * 60)

    # Freeze everything except decay parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    decay_params = []
    for name, param in model._kv2state_modules.named_parameters():
        if "log_decay" in name:
            param.requires_grad = True
            decay_params.append(param)
            logger.info(f"  Trainable: {name} = {torch.sigmoid(param.data).item():.4f}")

    num_decay_params = len(decay_params)
    logger.info(f"Trainable decay parameters: {num_decay_params}")

    if num_decay_params == 0:
        raise ValueError("No trainable decay parameters found. Set learnable_decay=True in KV2StateConfig.")

    optimizer = torch.optim.Adam(decay_params, lr=lr)

    # Linear warmup + cosine decay schedule
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # We need to compute teacher outputs layer-by-layer.
    # Strategy: for each batch, run the full model in eval mode to get hidden states,
    # then for each layer with streaming heads, compute teacher and student outputs.
    streaming_heads = head_classification.get_streaming_heads()
    mask = head_classification.mask
    num_kv_heads = mask.shape[1]
    model_config = model.config
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_q_heads = model_config.num_attention_heads
    q_per_kv = num_q_heads // num_kv_heads

    # Resolve rotary_emb (location varies across transformers versions:
    # model.model in 4.46+, per-layer attn in older, or elsewhere in latest)
    rotary_emb = None
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb"):
            rotary_emb = mod
            break
    if rotary_emb is None:
        raise RuntimeError(
            "Cannot find rotary_emb in model. Check transformers version."
        )

    # Group streaming heads by layer
    streaming_by_layer = {}
    for layer_idx, head_idx in streaming_heads:
        streaming_by_layer.setdefault(layer_idx, []).append(head_idx)

    data_iter = _get_streaming_data_iterator(
        tokenizer, dataset_name, dataset_subset, seq_len, batch_size,
    )

    losses = []
    model.eval()  # Keep model in eval mode for teacher; only decay params have grad

    pbar = tqdm(range(num_steps), desc="Stage 1 (MSE)")
    for step in pbar:
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = _get_streaming_data_iterator(
                tokenizer, dataset_name, dataset_subset, seq_len, batch_size,
            )
            input_ids = next(data_iter).to(device)

        state_cache.reset()

        # Get hidden states from the frozen model
        with torch.no_grad():
            hidden_outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states[i] = input to layer i (hidden_states[0] = embed output)
            all_hidden = hidden_outputs.hidden_states

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_heads_in_batch = 0

        for layer_idx, head_indices in streaming_by_layer.items():
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn

            # Input to this layer's attention
            hidden = all_hidden[layer_idx].detach()  # [B, T, hidden_size]
            bsz, T, _ = hidden.shape

            # Project Q, K, V
            with torch.no_grad():
                q = attn.q_proj(hidden).view(bsz, T, num_q_heads, head_dim).transpose(1, 2)
                k = attn.k_proj(hidden).view(bsz, T, num_kv_heads, head_dim).transpose(1, 2)
                v = attn.v_proj(hidden).view(bsz, T, num_kv_heads, head_dim).transpose(1, 2)

                # Apply RoPE
                position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, -1)
                cos, sin = rotary_emb(v, position_ids)
                q, k = _apply_rotary_pos_emb(q, k, cos, sin)

            for head_idx in head_indices:
                state_mod = model._kv2state_modules[f"layer{layer_idx}_head{head_idx}"]
                q_start = head_idx * q_per_kv

                # Teacher: standard softmax attention for this KV group
                with torch.no_grad():
                    teacher_k = k[:, head_idx:head_idx+1].expand(-1, q_per_kv, -1, -1)
                    teacher_v = v[:, head_idx:head_idx+1].expand(-1, q_per_kv, -1, -1)
                    teacher_q = q[:, q_start:q_start+q_per_kv]  # [B, q_per_kv, T, D]

                    scale = 1.0 / math.sqrt(head_dim)
                    attn_weights = torch.matmul(teacher_q, teacher_k.transpose(-2, -1)) * scale
                    causal = torch.triu(
                        torch.full((T, T), float("-inf"), device=device, dtype=hidden.dtype),
                        diagonal=1,
                    )
                    attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden.dtype)
                    teacher_out = torch.matmul(attn_weights, teacher_v)  # [B, q_per_kv, T, D]

                # Student: state attention (decay is differentiable)
                # Apply elu+1 feature map for non-negative linear attention
                head_k = _feature_map(k[:, head_idx])  # [B, T, D]
                head_v = v[:, head_idx]  # [B, T, D]

                student_outs = []
                for qi in range(q_per_kv):
                    head_q = _feature_map(q[:, q_start + qi])  # [B, T, D]
                    out, _, _ = state_mod.parallel_forward(head_q, head_k, head_v)
                    student_outs.append(out)

                student_out = torch.stack(student_outs, dim=1)  # [B, q_per_kv, T, D]

                # MSE loss
                head_loss = F.mse_loss(student_out, teacher_out.detach())
                total_loss = total_loss + head_loss
                num_heads_in_batch += 1

        if num_heads_in_batch > 0:
            avg_loss = total_loss / num_heads_in_batch
        else:
            avg_loss = total_loss

        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(decay_params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = avg_loss.item()
        losses.append(loss_val)

        if step % log_interval == 0 or step == num_steps - 1:
            # Sample some decay values
            sample_decays = []
            for p in decay_params[:5]:
                sample_decays.append(f"{torch.sigmoid(p.data).item():.4f}")
            decay_str = ", ".join(sample_decays)
            if len(decay_params) > 5:
                decay_str += ", ..."
            pbar.set_postfix({
                "loss": f"{loss_val:.6f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                "decays": decay_str,
            })
            logger.info(
                f"Step {step}/{num_steps} | Loss: {loss_val:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | Decays: [{decay_str}]"
            )

    # Collect final decay values
    final_decays = {}
    for name, param in model._kv2state_modules.named_parameters():
        if "log_decay" in name:
            final_decays[name] = torch.sigmoid(param.data).item()

    logger.info("Stage 1 complete.")
    logger.info(f"Final decay values: {final_decays}")

    return {
        "losses": losses,
        "final_decays": final_decays,
        "num_steps": num_steps,
    }


def calibrate_stage2(
    model: nn.Module,
    tokenizer,
    state_cache,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: Optional[list[str]] = None,
    dataset_name: str = "DKYoon/SlimPajama-6B",
    dataset_subset: str = "train",
    lr: float = 2e-4,
    num_steps: int = 2000,
    batch_size: int = 2,
    seq_len: int = 256,
    warmup_steps: int = 100,
    log_interval: int = 100,
    device: str = "cuda",
) -> dict:
    """Stage 2: End-to-end LoRA fine-tuning with cross-entropy LM loss.

    Adds LoRA adapters to q_proj/v_proj (configurable), freezes calibrated decay
    params, and trains with standard next-token prediction loss on SlimPajama.

    Args:
        model: Patched KV2State model with calibrated decay (from Stage 1).
        tokenizer: Tokenizer for the model.
        state_cache: StateCache (reset each batch).
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_target_modules: Which modules to apply LoRA to. Default: ["q_proj", "v_proj"].
        dataset_name: HuggingFace dataset for training data.
        dataset_subset: Dataset split.
        lr: Learning rate for LoRA parameters.
        num_steps: Number of training steps.
        batch_size: Batch size.
        seq_len: Sequence length per sample.
        warmup_steps: Linear warmup steps.
        log_interval: Steps between logging.
        device: Device to train on.

    Returns:
        Dict with training stats: losses, etc.
    """
    from peft import LoraConfig, get_peft_model

    logger.info("=" * 60)
    logger.info("Stage 2: End-to-end LoRA fine-tuning (cross-entropy)")
    logger.info("=" * 60)

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    # Freeze all parameters first (including calibrated decay)
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Collect trainable params (LoRA only — decay is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters (LoRA): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    data_iter = _get_streaming_data_iterator(
        tokenizer, dataset_name, dataset_subset, seq_len, batch_size,
    )

    losses = []
    model.train()

    pbar = tqdm(range(num_steps), desc="Stage 2 (LoRA)")
    for step in pbar:
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = _get_streaming_data_iterator(
                tokenizer, dataset_name, dataset_subset, seq_len, batch_size,
            )
            input_ids = next(data_iter).to(device)

        state_cache.reset()

        # Standard causal LM forward: labels = input_ids shifted
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_interval == 0 or step == num_steps - 1:
            ppl = math.exp(min(loss_val, 20.0))  # Clamp to avoid overflow
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "ppl": f"{ppl:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            })
            logger.info(
                f"Step {step}/{num_steps} | Loss: {loss_val:.4f} | "
                f"PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    model.eval()

    logger.info("Stage 2 complete.")
    logger.info(f"Final loss: {losses[-1]:.4f}")

    return {
        "losses": losses,
        "num_steps": num_steps,
        "model": model,  # Returns PEFT-wrapped model
    }
