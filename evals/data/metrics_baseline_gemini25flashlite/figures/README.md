# Figure Generation Workflow

This directory follows the **Churkin Protocol** for separating Raw Assets from Production Assets.

## Folder Structure

```
figures/
├── raw/              # Code-generated figures (safe to overwrite)
│   ├── figure1/
│   │   ├── figure1.pdf
│   │   └── figure1.svg
│   ├── figure2/
│   │   ├── figure2.pdf
│   │   └── figure2.svg
│   └── figure3/
│       ├── figure3.pdf
│       └── figure3.svg
└── final/            # Manually edited production figures (protected)
    ├── figure1/
    │   ├── figure1.pdf
    │   └── figure1.svg
    ├── figure2/
    │   ├── figure2.pdf
    │   └── figure2.svg
    └── figure3/
        ├── figure3.pdf
        └── figure3.svg
```

## Workflow

### 1. Generate Raw Figures (Code)

Run the figure generation script:
```bash
python -m evals.src.paper_figures.generate_all
```

**Result:** All figures are saved to `raw/` folder. These files can be safely overwritten anytime.

### 2. Manual Refinement (Human)

1. **Open** the raw figure in Inkscape (or Adobe Illustrator):
   - `raw/figure1/figure1.svg` (preferred for editing)
   - Or `raw/figure1/figure1.pdf`

2. **Edit** manually:
   - Adjust font sizes
   - Nudge labels
   - Fine-tune spacing
   - Remove white background boxes
   - Verify font sizes match 8pt reference when scaled

3. **Export** to final folder:
   - Save as `final/figure1/figure1.pdf` (for LaTeX)
   - Save as `final/figure1/figure1.svg` (backup)

### 3. Use in LaTeX (Production)

In your LaTeX file (`paper.tex`), reference the **final** folder:

```latex
\includegraphics[width=\columnwidth]{evals/data/metrics/figures/final/figure1/figure1.pdf}
```

## Safety Benefits

- **Code → raw/**: Scripts can overwrite raw files without affecting production
- **Human → final/**: Manual edits are protected from code regeneration
- **LaTeX → final/**: Always uses the refined, production-ready versions

## Best Practices

1. **Never edit files in `raw/`** - they will be overwritten
2. **Always edit and save to `final/`** - protected from code changes
3. **Keep both PDF and SVG** in final/ - PDF for LaTeX, SVG for future edits
4. **Version control**: Consider committing `final/` but not `raw/` (or use `.gitignore`)




