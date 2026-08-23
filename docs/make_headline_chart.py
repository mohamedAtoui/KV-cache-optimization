"""Headline chart: quality cost against KV cache compression.

Numbers are the Llama-3.2-1B-Instruct / WikiText-2 run recorded in
docs/DIARY.md (Day 20) and reproduced by
streaming_attention/notebooks/03-kv-bench-modal.ipynb.

    uv run --with matplotlib python docs/make_headline_chart.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# strategy, compression x, delta perplexity vs FullKV (11.15)
POINTS = [
    ("INT8-all", 1.9, 0.00),
    ("Stratigraphic", 2.2, 0.04),
    ("INT4-all", 3.6, 2.22),
    ("SnapKV (50%)", 2.0, 6.79),
    ("H2O (50%)", 2.0, 96.57),
]
HERO = "Stratigraphic"

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e3e2df", hero="#2a78d6", other="#8b8a85"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#33322f", hero="#3987e5", other="#8b8a85"),
}

# (dx, dy, horizontal alignment)
LABEL_OFFSETS = {
    "INT8-all": (0, 20, "center"),
    "Stratigraphic": (16, 2, "left"),
    "INT4-all": (0, 18, "center"),
    "SnapKV (50%)": (14, 4, "left"),
    "H2O (50%)": (14, 0, "left"),
}


def render(theme: str, path: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for name, comp, dppl in POINTS:
        hero = name == HERO
        ax.scatter([comp], [max(dppl, 0.01)], s=210 if hero else 120,
                   color=t["hero"] if hero else t["other"],
                   edgecolors=t["surface"], linewidths=2,
                   zorder=4 if hero else 3)
        dx, dy, ha = LABEL_OFFSETS[name]
        ax.annotate(f"{name}\n+{dppl:.2f} PPL at {comp}x", (comp, max(dppl, 0.01)),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha=ha, va="center",
                    color=t["primary"] if hero else t["secondary"],
                    fontweight="bold" if hero else "normal")

    ax.set_yscale("log")
    ax.set_ylim(0.008, 400)
    ax.set_xlim(1.5, 4.4)
    ax.set_xlabel("KV cache compression  (higher is better →)", fontsize=10, color=t["secondary"])
    ax.set_ylabel("perplexity added vs full cache  (lower is better ↓)", fontsize=10,
                  color=t["secondary"])
    ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_xticklabels(["2.0x", "2.5x", "3.0x", "3.5x", "4.0x"])
    ax.tick_params(colors=t["secondary"], labelsize=9)
    ax.grid(True, which="major", color=t["grid"], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])

    ax.set_title("Stratigraphic: 2.2x smaller KV cache for +0.04 perplexity",
                 fontsize=13, color=t["primary"], loc="left", pad=14, fontweight="bold")
    ax.annotate("Llama-3.2-1B-Instruct, WikiText-2, A100. Baseline perplexity 11.15. "
                "INT8 measured +0.00 and is drawn at the axis floor.",
                xy=(0, -0.155), xycoords="axes fraction", fontsize=8.5, color=t["secondary"])

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render("light", "docs/assets/headline-light.png")
    render("dark", "docs/assets/headline-dark.png")
