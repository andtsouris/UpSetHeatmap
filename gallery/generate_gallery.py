"""
UpSetHeatmap Gallery
====================
Run this script to generate example plots showcasing all major features.
Plots are saved as PNG files in the same directory as this script.

Usage:
    python gallery/generate_gallery.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from upsetheatmap import UpSet, generate_samples

OUT = os.path.dirname(os.path.abspath(__file__))

# ── shared datasets ──────────────────────────────────────────────────────────
data_std   = generate_samples(seed=0,  n_samples=1000, n_categories=4, n_groups=5)
data_large = generate_samples(seed=1,  n_samples=3000, n_categories=5, n_groups=6)
data_small = generate_samples(seed=42, n_samples=200,  n_categories=3, n_groups=3)


def save(fig, name):
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")


# ── 01  Basic horizontal (default) ───────────────────────────────────────────
print("01  Basic horizontal")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality")
upset.plot(fig)
fig.suptitle("01 · Basic horizontal layout", fontsize=13, fontweight="bold")
save(fig, "01_basic_horizontal")


# ── 02  Vertical orientation ──────────────────────────────────────────────────
print("02  Vertical orientation")
fig = plt.figure(figsize=(8, 12))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              orientation="vertical")
upset.plot(fig)
fig.suptitle("02 · Vertical orientation", fontsize=13, fontweight="bold")
save(fig, "02_vertical")


# ── 03  Show counts & percentages ─────────────────────────────────────────────
print("03  Show counts & percentages")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              show_counts=True, show_percentages=True)
upset.plot(fig)
fig.suptitle("03 · show_counts + show_percentages", fontsize=13, fontweight="bold")
save(fig, "03_counts_percentages")


# ── 04  Heatmap normalisation — fraction ─────────────────────────────────────
print("04  Heatmap normalisation — fraction")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              heatmap_normalize="fraction", heatmap_cmap="YlOrRd")
upset.plot(fig)
fig.suptitle("04 · Heatmap: fraction normalisation", fontsize=13, fontweight="bold")
save(fig, "04_heatmap_fraction")


# ── 05  Heatmap normalisation — z-score ──────────────────────────────────────
print("05  Heatmap normalisation — z-score")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              heatmap_normalize="zscore", heatmap_cmap="RdBu_r")
upset.plot(fig)
fig.suptitle("05 · Heatmap: z-score normalisation", fontsize=13, fontweight="bold")
save(fig, "05_heatmap_zscore")


# ── 06  Heatmap colour palettes ───────────────────────────────────────────────
print("06  Heatmap colour palettes (2×2 grid)")
cmaps = [("viridis", None), ("plasma", None),
         ("coolwarm", "zscore"), ("Blues", "fraction")]
fig, axes = plt.subplots(2, 2, figsize=(18, 10),
                         gridspec_kw={"hspace": 0.35, "wspace": 0.35})
fig.suptitle("06 · Heatmap colour palettes", fontsize=14, fontweight="bold")
for ax, (cmap, norm) in zip(axes.flat, cmaps):
    sub = plt.figure(figsize=(9, 4.5))
    upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
                  heatmap_cmap=cmap, heatmap_normalize=norm)
    axes_dict = upset.plot(sub)
    label = f"cmap='{cmap}'" + (f", normalize='{norm}'" if norm else "")
    sub.suptitle(label, fontsize=10)
    path = os.path.join(OUT, f"06_palette_{cmap}.png")
    sub.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(sub)
plt.close(fig)
print("  saved 06_palette_*.png")


# ── 07  Style subsets (highlight specific intersections) ─────────────────────
print("07  style_subsets")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality")
upset.style_subsets(present="cat0",
                    facecolor="steelblue", label="contains cat0")
upset.style_subsets(present="cat1", absent=["cat0", "cat2", "cat3"],
                    facecolor="tomato", label="cat1 only")
upset.style_subsets(min_degree=3,
                    edgecolor="gold", linewidth=2, label="degree ≥ 3")
upset.plot(fig)
fig.suptitle("07 · style_subsets — highlight intersections", fontsize=13, fontweight="bold")
save(fig, "07_style_subsets")


# ── 08  Stacked bars (group breakdown) ────────────────────────────────────────
print("08  Stacked bars")
fig = plt.figure(figsize=(13, 8))
tab10 = plt.cm.tab10.colors
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              show_counts=True)
upset.add_stacked_bars(by="group", colors=tab10,
                       title="Group breakdown", elements=3)
upset.plot(fig)
fig.suptitle("08 · add_stacked_bars — per-group breakdown", fontsize=13, fontweight="bold")
save(fig, "08_stacked_bars")


# ── 09  Filter by degree ──────────────────────────────────────────────────────
print("09  Filter by degree")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_large, subset_size="count", sort_by="cardinality",
              min_degree=2, max_degree=3)
upset.plot(fig)
fig.suptitle("09 · Filter: min_degree=2, max_degree=3", fontsize=13, fontweight="bold")
save(fig, "09_filter_degree")


# ── 10  Filter by subset size + top-N ─────────────────────────────────────────
print("10  Filter by subset size + top-N rank")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_large, subset_size="count", sort_by="cardinality",
              min_subset_size=50, max_subset_rank=8)
upset.plot(fig)
fig.suptitle("10 · Filter: min_subset_size=50, max_subset_rank=8",
             fontsize=13, fontweight="bold")
save(fig, "10_filter_size_rank")


# ── 11  Custom group order ────────────────────────────────────────────────────
print("11  Custom group order")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              sort_groups_by="custom",
              group_order=["group4", "group3", "group2", "group1", "group0"])
upset.plot(fig)
fig.suptitle("11 · Custom group order (reversed)", fontsize=13, fontweight="bold")
save(fig, "11_custom_group_order")


# ── 12  Groups sorted by size ─────────────────────────────────────────────────
print("12  Groups sorted by count")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              sort_groups_by="count",
              heatmap_normalize="fraction", heatmap_cmap="YlGnBu")
upset.plot(fig)
fig.suptitle("12 · sort_groups_by='count' + fraction normalisation",
             fontsize=13, fontweight="bold")
save(fig, "12_groups_sorted_by_count")


# ── 13  Dark background + custom facecolor ────────────────────────────────────
print("13  Dark theme")
with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(13, 6), facecolor="#1c1c1c")
    upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
                  facecolor="white", shading_color="#2e2e2e",
                  other_dots_color="#555555",
                  heatmap_cmap="magma")
    upset.plot(fig)
    fig.suptitle("13 · Dark theme with magma colormap",
                 fontsize=13, fontweight="bold", color="white")
    save(fig, "13_dark_theme")


# ── 14  include_empty_subsets ─────────────────────────────────────────────────
print("14  include_empty_subsets")
fig = plt.figure(figsize=(13, 6))
upset = UpSet(data_small, subset_size="count", sort_by="degree",
              include_empty_subsets=True, show_counts=True,
              heatmap_cmap="PuBuGn")
upset.plot(fig)
fig.suptitle("14 · include_empty_subsets=True (sort_by='degree')",
             fontsize=13, fontweight="bold")
save(fig, "14_include_empty_subsets")


# ── 15  Vertical + z-score + style_subsets ────────────────────────────────────
print("15  Vertical + z-score + style_subsets")
fig = plt.figure(figsize=(9, 13))
upset = UpSet(data_std, subset_size="count", sort_by="cardinality",
              orientation="vertical",
              heatmap_normalize="zscore", heatmap_cmap="PiYG")
upset.style_subsets(present="cat2", facecolor="#e07b39", label="contains cat2")
upset.plot(fig)
fig.suptitle("15 · Vertical + z-score + style_subsets",
             fontsize=13, fontweight="bold")
save(fig, "15_vertical_zscore_style")


# ── 16  Large dataset — 5 categories ─────────────────────────────────────────
print("16  Large dataset (5 categories, 6 groups)")
fig = plt.figure(figsize=(16, 7))
upset = UpSet(data_large, subset_size="count", sort_by="cardinality",
              max_subset_rank=16,
              heatmap_normalize="fraction", heatmap_cmap="YlOrRd",
              show_counts=True)
upset.plot(fig)
fig.suptitle("16 · Large dataset: 5 categories, 6 groups, top-16 intersections",
             fontsize=13, fontweight="bold")
save(fig, "16_large_dataset")


print("\nDone — all plots saved to", OUT)
