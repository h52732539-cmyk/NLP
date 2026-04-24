#!/usr/bin/env python3
"""
Package the NLP report into an Overleaf-ready .zip file.
Run from NLP/ directory:
    python report/build_zip.py
"""
import os
import shutil
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
NLP_DIR = Path(__file__).parent.parent   # NLP/
REPORT_DIR = NLP_DIR / "report"
FIGURES_SRC = NLP_DIR / "results" / "figures"
OUTPUT_ZIP = NLP_DIR / "report" / "overleaf_upload.zip"

# Figures we want to include (subset for the paper)
FIGURE_NAMES = [
    "roc_no_thinking.pdf",
    "roc_thinking.pdf",
    "distributions_no_thinking_bge-base.pdf",
    "auc_heatmap_no_thinking.pdf",
    "auc_heatmap_thinking.pdf",
    "short_vs_long_bge-base.pdf",
    "thinking_ablation_bge-base.pdf",
    "failure_causes_no_thinking.pdf",
    "failure_causes_thinking.pdf",
    "delta_auc_boxplot.pdf",
    "sim_vs_length_no_thinking_bge-base.pdf",
    "delta_auc_maxsim_no_thinking.pdf",
    "delta_auc_maxsim_thinking.pdf",
    "roc_overlay_truthfulqa_bge-base.pdf",
    "roc_overlay_natural_questions_bge-base.pdf",
    "dist_comparison_maxsim_thinking_bge-base.pdf",
]


def build_zip():
    print(f"Building Overleaf zip → {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. main.tex
        main_tex = REPORT_DIR / "main.tex"
        zf.write(main_tex, "main.tex")
        print(f"  + main.tex")

        # 2. references.bib
        bib = REPORT_DIR / "references.bib"
        zf.write(bib, "references.bib")
        print(f"  + references.bib")

        # 3. Figures
        missing = []
        for fname in FIGURE_NAMES:
            src = FIGURES_SRC / fname
            if src.exists():
                zf.write(src, f"figures/{fname}")
                print(f"  + figures/{fname}")
            else:
                missing.append(fname)

        if missing:
            print(f"\n  WARNING: {len(missing)} figure(s) not found:")
            for f in missing:
                print(f"    - {f}")

    print(f"\nDone. Zip size: {OUTPUT_ZIP.stat().st_size / 1024:.1f} KB")
    print(f"Upload to: https://www.overleaf.com/  →  New Project → Upload Project → select the .zip")


if __name__ == "__main__":
    build_zip()
