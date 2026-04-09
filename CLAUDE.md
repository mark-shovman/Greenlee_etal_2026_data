# CLAUDE.md — AI Assistant Guide for Greenlee et al. 2026

This file documents the codebase structure, conventions, and workflows for AI assistants working in this repository.

## Project Overview

This is a **scientific data repository** accompanying the paper:

> Greenlee M, Keil E, Plank T, Jägle H, Lee Chen D M, Lu I, Vorobey A, Shovman M, Kashchenevsky A.
> **Accurate AI-assisted binocular limbus eyetracking in participants performing visually guided eye-movement tasks.**

The repository contains raw eye-tracking data from three experiments and Python analysis scripts that reproduce the paper's figures and statistics.

## Repository Structure

```
Greenlee_etal_2026_data/
├── Code/
│   ├── analyse_accuracy.py   # Accuracy experiment analysis (complete)
│   ├── analyse_luminance.py  # Luminance experiment analysis (placeholder)
│   ├── analyse_vergence.py   # Vergence experiment analysis (placeholder)
│   └── utils.py              # Shared utilities (E|U percentile framework)
├── Data/
│   ├── 1_accuracy/           # 142 CSVs, ~44MB — fixation grid accuracy data
│   ├── 2_luminance/          # 103 CSVs, ~33MB — gaze under luminance variation
│   └── 3_vergence/           # 136 CSVs, ~95MB — binocular vergence data
├── CLAUDE.md                 # This file
├── LICENSE                   # MIT
├── README.md                 # User-facing documentation
└── requirements.txt          # Pinned Python dependencies (UTF-16LE encoded)
```

## Development Branch

All development work goes on branch `claude/add-claude-documentation-233R3`. Push with:
```bash
git push -u origin claude/add-claude-documentation-233R3
```

## Running the Code

All scripts must be run from the **repository root** (not from inside `Code/`), because they use relative paths like `Data/1_accuracy/` and imports like `from Code.utils import ...`.

```bash
# Install dependencies
pip install -r requirements.txt

# Run analyses
python Code/analyse_accuracy.py
python Code/analyse_luminance.py
python Code/analyse_vergence.py
```

The scripts use `# %%` cell markers, making them compatible with both standard Python execution and Jupyter-style interactive runners (e.g., VS Code Python Interactive, Spyder).

## Data Conventions

### File Naming

| Dataset | Pattern | Example |
|---------|---------|---------|
| Accuracy | `{pid}_grid3_{eye}.csv` | `p042_grid3_left.csv` |
| Luminance | `{pid}_{eye}.csv` | `p007_right.csv` |
| Vergence | `{pid}_vergence_{session}.csv` | `p031_vergence_2.csv` |

- `{pid}` is zero-padded (e.g., `p000`–`p057`)
- `{eye}` is `left` or `right`
- Vergence `{session}` is `1`, `2`, or `3`

### Units

- Gaze/stimulus positions: degrees of visual angle (dva) or degrees
- Pupil diameter: millimeters
- Vergence limbus/gaze vectors: millimeters, in marker coordinate system (origin = marker center, X right, Y down, Z away from participant)
- Timestamps: seconds

### Key Columns by Dataset

**1_accuracy:**
- `deviation_from_stimulus_{x|y}_dva`: gaze error (the primary accuracy metric)
- `stimulus_{x|y}_deg`: where on screen the fixation target was
- `kappa_{x|y}_deg`: per-eye visual-to-optical axis correction (constant)
- `is_fixation_outlier`: boolean — exclude these rows for clean analysis
- `participant_in_training_set`: boolean — flag for train/test split reporting

**2_luminance:**
- `{x|y}_dva`: Fick's angle gaze position relative to head
- `pupil_diameter_mm`: pupil size
- `luminance`: categorical `'bright'` or `'dark'`
- `fixation_id`: I-VT fixation segmentation ID

**3_vergence:**
- `ic{X|Y|Z}_{left|right}`: limbus center coordinates (mm)
- `ig{X|Y|Z}_{left|right}`: gaze vector (points **into** the eye by convention)
- `stimulus_order_from_viewers`: 1 (nearest) to 6 (farthest)
- `reference_fixation_id`: non-empty rows belong to the vergence reference fixation

## Code Patterns and Conventions

### Data Loading Pattern

All three scripts follow the same idiom — iterate over a data directory, parse the filename for metadata, load the CSV, attach parsed metadata as columns, and concatenate:

```python
df_all = None
for fname in os.listdir("Data/1_accuracy"):
    pid, grid, eye = fname.split(".")[0].split("_")
    df = pd.read_csv(f"Data/1_accuracy/{fname}")
    df["participant"] = pid
    df["grid"] = grid
    df["eye"] = eye
    df_all = pd.concat((df_all, df)) if df_all is not None else df
```

Keep this pattern consistent when extending the luminance and vergence scripts.

### E|U Percentile Framework

`utils.calculate_eu_percentiles()` implements the Raju et al. (2025 ETRA) framework for reporting eye-tracker accuracy at both individual (E) and population (U) levels:

```python
eu = calculate_eu_percentiles(
    df,
    uid_col="participant",    # grouping column
    data_col="accuracy",      # metric column
    e_percentiles=[10, 25, 50, 75, 90],
    u_percentiles=[10, 25, 50, 75, 90],
)
# Returns a pivot DataFrame: rows=U percentile, columns=E percentile
```

- **E percentile**: within-participant distribution of the metric (e.g., E50 = median per participant)
- **U percentile**: across-participant distribution of each E percentile (e.g., U95–E95 = worst-case across population)

### Outlier Handling

In the accuracy analysis, outlier fixations are removed with a threshold of 0.5 on the `is_fixation_outlier` column (which is encoded as a float after `groupby.mean()`):

```python
fix_clean = fix[fix["is_fixation_outlier"] < 0.5]
```

Apply the same logic in luminance/vergence analyses when aggregating to fixation-level data.

### Visualization Conventions

- Heatmaps use `seaborn.heatmap` with `annot=True, fmt=".2f", square=True, cbar=False`
- Accuracy heatmap color scale: `vmin=0.25, vmax=1.33` (degrees)
- Precision heatmap color scale: `vmin=0.12, vmax=0.29` (degrees)
- E|U heatmaps use `cmap="flare_r"`
- Always call `ax.invert_yaxis()` on spatial heatmaps so that Y increases upward (screen convention)
- Run scripts from root; plots are shown interactively via `plt.show()`

## Implementation Status

| Script | Status | Notes |
|--------|--------|-------|
| `analyse_accuracy.py` | Complete | Data loading, fixation aggregation, heatmaps, E|U table |
| `analyse_luminance.py` | Placeholder | Data loading only; analysis not yet implemented |
| `analyse_vergence.py` | Placeholder | Data loading only; analysis not yet implemented |
| `utils.py` | Complete | `calculate_eu_percentiles` ready for reuse |

The `accuracy` and `precision` columns in `analyse_accuracy.py` are set to `None` (lines 40–41) and marked with `# TODO` — these need to be computed from the deviation columns before the heatmaps and E|U tables will contain real values.

## Dependencies

Dependencies are pinned in `requirements.txt` (UTF-16LE encoded — read with appropriate encoding if parsing programmatically). Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.3.5 | Numerical arrays, percentile calculation |
| pandas | 2.3.3 | DataFrames, CSV loading, groupby |
| scipy | 1.16.3 | Statistical tests |
| statsmodels | 0.14.6 | Statistical modeling |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| black | 25.12.0 | Code formatting |

## Testing and CI

There is no automated test suite or CI/CD pipeline. This is a research data publication, not production software. Manual verification is done by running the analysis scripts and inspecting the generated plots.

## Code Style

- Formatter: `black` (installed via requirements.txt; use `black Code/` to format)
- No strict type annotations required
- Cell-marker style (`# %%`) is intentional — preserves Jupyter-style interactive workflow
- Keep scripts runnable both interactively (cell-by-cell) and as whole-file scripts

## Important Notes for AI Assistants

1. **Run from root**: All relative paths (`Data/...`, `Code/utils`) assume the working directory is the repository root.
2. **Data is read-only**: Never modify files under `Data/`. Scripts only read from there.
3. **Incomplete scripts**: `analyse_luminance.py` and `analyse_vergence.py` are intentional stubs. When implementing them, follow the patterns in `analyse_accuracy.py`.
4. **No tests**: Do not add a test framework unless explicitly requested. Validate changes by running the scripts manually.
5. **Paper in review**: Do not update the citation section in `README.md` or make claims about publication status without confirmation from the authors.
6. **Large data files**: The `Data/` directory is ~172MB. Avoid loading all datasets simultaneously in memory when possible.
