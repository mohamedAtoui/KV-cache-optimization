

# Week 1:

citation and more details could be founded here: https://fr.overleaf.com/read/jjfgydhthgdc#ca2197

## 📅 Day 1: October 14th – Baseline and Optimization

I began by establishing the **MHA (Multi-Head Attention)** baseline from the foundational *“Attention Is All You Need”* paper.  
Then I implemented **MQA (Multi-Query Attention)**, which optimizes inference speed by sharing a single **Key (K)** and **Value (V)** projection across all attention heads.  

This approach:
- Sharply reduces **memory bandwidth** requirements.  
- Lowers **decoding latency**.  
- Highlights the critical **speed–quality trade-off** for large language models (LLMs).

---

## 📅 Day 2: October 15th – The Optimal Compromise

Today I focused on **GQA (Grouped-Query Attention)** — a balanced compromise between MHA and MQA.  
**GQA** maintains all **Query (Q)** heads separately for expressiveness but shares **K** and **V** projections across **G** groups (where `1 < G < H`).

This strategy:
- Preserves much of **MQA’s speed advantage**.  
- Recovers **model quality** lost in MQA due to excessive sharing.  

I now have **three distinct attention variants** (MHA, MQA, GQA) ready for comparison in the project.


## 📅 Day 3: October 17th – Breaking the Quadratic Barrier

Today I explored **Reformer** — a breakthrough in making Transformers efficient for long sequences.  
The core innovation is **Locality-Sensitive Hashing (LSH) attention**, which reduces the prohibitive O(L²) complexity to O(L log L).

### The key insights:
- **Attention is naturally sparse** — queries only need to attend to similar keys
- **LSH groups similar items** into hash buckets, computing attention only within buckets
- **Reversible layers** eliminate the need to store N× activations during backprop
- **Chunking** handles memory-hungry feed-forward layers

### This enables:
- Training on **64K token sequences** (vs ~512 in standard Transformers)
- Orders of magnitude **memory savings** without sacrificing quality
- Accessible training on consumer hardware instead of massive clusters

# Week 2:

## 📅 Day 4: October 22nd – Dataset Preparation & Tokenization Pipeline

Today I established the complete data preprocessing infrastructure for the project.
Selected WikiText-2 (12MB) for debugging and WikiText-103 (500MB) for experiments, implementing a full tokenization pipeline with the GPT-2 BPE tokenizer (50,257 vocab, 512 token max length).

### Key achievements:

- Analyzed sequence length distributions to set optimal max length
- Implemented attention masking to handle padding correctly
- Validated batch generation produces correct tensor shapes
- Saved processed datasets with configuration for reproducibility

Both datasets are now tokenized, validated, and ready for Week 3's Transformer model training.


## Week 3:

### 📅 Day 5: October 27th – Multi-Head Attention Implementation Complete

Today I completed the full baseline Multi-Head Attention (MHA) Transformer implementation. Built a complete encoder-decoder architecture (6 layers each, 8 attention heads, ~177M parameters) with all core components and training infrastructure.

#### Key achievements:

- Implemented scaled dot-product attention with causal and padding masks
- Built complete transformer with positional encoding (sinusoidal/learned options)
- Developed training pipeline with TensorBoard logging and checkpoint management
- Created WikiText data loader with dynamic batching
- Verified attention weights sum to 1.0 and all shape transformations are correct
- Set up Google Colab notebook for GPU training with Drive integration
- Wrote comprehensive technical report and documentation

### 📅 Day 6: October 31st – Project Consolidation & Documentation Complete

Today I consolidated the entire AttentionHeads project into a complete, production-ready package with comprehensive documentation and training infrastructure.

#### Key achievements:

- Finalized complete MHA implementation with all core transformer components
- Developed multiple training notebooks for different scenarios:
  - Colab notebook for cloud GPU training with WikiText dataset
  - TinyStories dataset integration notebook (prepared for future experiments)
  - Transformer v2 and v3 training pipelines
- Set up complete project structure with proper packaging (setup.py, requirements.txt)
- Configured version control and gitignore for ML artifacts

#### Current status:

The project now has a robust foundation for experimentation. **WikiText-2/103 datasets** are currently used for initial training and validation. The **TinyStories dataset** integration is prepared for future experiments, offering a simpler narrative structure that may provide better insights into attention mechanism behavior on coherent story generation tasks.

Next steps involve running full training experiments on TinyStories datasets.

---

## Week 4:
### 📅 Day 7: November 4th – Changing Dataset to use Tiny Stories:

- After a failed expriement with WikiText dataset, which was caused because of the complexity of dataset (Advanced language, different writing styles,...), I changed everything to use TinyStories dataset.
eventhough I noticed a weird behaviour of increase in loss and perplexity, which I think goes bakc to the parameters I adjusted, specifically the learning rate.
- Other thing I focused on is the code base which defines differnet parts of the Transformer, I solved a compatiblity issue of pytorch library features with computing which I use (L4 GPU).


# Week 5:
### 📅 Day 8: November 11th:
After a long period because of other attachements, I started again and I managed to train correctly my Transformer using TinyStories Dataset with Multi Head Attention,
updating next parameters and helped in correting my training and reducing training time:
 - num_layers: 4 (reduced from 8)
 - learning_rate: 5e-5 (reduced from 1e-3 )
 - gradient_accumulation_steps: 4 (increased from 2)
 - effective_batch_size: 256 (doubled from 128)
 - warmup_steps: 600 (doubled from 300)
 - gradient_clip: 0.5 (tighter from 1.0)
 - compile_model: true (enabled PyTorch compilation)



 
# week 6:

### Day 9: November 19th:
I managed yesterday to finish training LLM which uses MQA attention head strcutre, the result is good from first try and that may happened because I used simialire architecture to previous MHA with only single difference where all 8 heads share same KV:
which means:

-   Q projection: 256 → 256  (8 heads × 32 dims each)
-   K projection: 256 → 32   (1 head  × 32 dims) ← SHARED across all 8 Q heads
-   V projection: 256 → 32   (1 head  × 32 dims) ← SHARED across all 8 Q heads
-   Output:      256 → 256

### Day 10: November 22nd:
I managed today to compare between MQA and MAH models, first of all due to the small size of both models which is 16M parameters there was not that noticable Optimization in time and memory,
but it was alligning with what MQA paper says () where I measured a speed up of 1.02x knowing that I have 256 token sequences (short) so in 2048 tokens it is expected to be: 1.5-2.5×

KV-Cache = 2 × num_layers × seq_length × num_kv_heads × head_dim × bytes_per_param
Our case: 256 token seqs 
KV-Cache(MHA) = 2 × 4 × 256 × **8** × 64 × 2 = 2.1 MB 
KV-Cache(MQA) = 2 × 4 × 256 × **1** × 64 × 2 = 0.26 MB
--> KV-Cache(MHA) = 8 x KV-Cache(MQA)

and based on Shazeer (2019) MQA Paper:  

Speedup = 1 / (a + (1-a)/H):
  - a = fraction of time spent on non-attention compute
  - H = number of heads (reduction factor for MQA)

At small scale with short sequences, a = 0.95 (95% compute, 5% memory):
Speedup = 1 / (0.95 + 0.05/8) = 1 / 0.956 = 1.04×

At larger scale with longer sequences, a = 0.5 (50% compute, 50% memory):
Speedup = 1 / (0.5 + 0.5/8) = 1 / 0.5625 = 1.78×


intresting thing I foudn today which is: Pythia-1B which is an experimental LLMs in different sizes (70M-12B) where I will change its head of attention (by default is MHA)
into MQA and GQA using **uptraining** as  GQA paper (Ainslie et al., 2023) demonstrated.


# week 7:


### Day 11 November 24th:
After reading Pythia paper I decided to continue 

Today's I finished setting-up GQA Attention Head which has next strcutre:

- Query tensor of shape (..., seq_len_q, d_k)
- key: Key tensor of shape (..., seq_len_k, d_k)
- value: Value tensor of shape (..., seq_len_v, d_k)

so if we comapre it to MHA and MQA:
- MHA: h query heads, h KV heads (each query head has its own K,V)
- MQA: h query heads, 1 KV head (all query heads share K,V)
- GQA: h query heads, g KV heads where 1 < g < h (groups of query heads share K,V)

and after a quick searhc it's being used more than MHA and MQA in real world LLM's (LLama 2, Mistral 7b,...)


### Day 12 November 25th:
Today I managed to train GQA Attention head with 4 Attention head, same architecture as MHA but with half attention heads, next thing is comparing:
MHA, MQA, GQA in different aspects: quality of responses, inference time, memory usage (my approach and forcast with a big number of parameters).




