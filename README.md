# Accurate AI-assisted binocular limbus eye tracking in participants performing visually guided eye-movement tasks

This repository contains the data and analysis code for the paper:

> Greenlee M, Keil E, Plank T, Jägle H, Lee Chen D M, Lu I, Vorobey A, Shovman M, Kashchenevsky A.
>
> **Accurate AI-assisted binocular limbus eye tracking in participants performing visually guided eye-movement tasks.**

The paper is currently in review, and the details will be updated here if and when it is published.

The scripts in `Code/` reproduce the paper's Figures 4–15, Table 1, and the statistics quoted in
the Results from the data in `Data/`. See [Errata](#errata) for the places where the submitted
text differs from what the data give.

## Repository Structure

```text
.
├── Code/                   # Analysis and plotting scripts
│   ├── analyse_accuracy.py    # Grid fixation: Figs 4-9, Table 1
│   ├── analyse_luminance.py   # Background luminance change: Figs 10-12
│   ├── analyse_vergence.py    # Visual search: Figs 13-15
│   └── utils.py
├── Data/
│   ├── 1_accuracy/
│   ├── 2_luminance/
│   └── 3_vergence/
├── LICENSE
├── README.md
└── requirements.txt        # Python dependencies
```

## Getting Started

1. Install the dependencies (Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analyses. Each script prints the numbers reported in the paper and saves its figures to
   `Figures/` (add `--show` to also display them):
   ```bash
   python Code/analyse_accuracy.py
   python Code/analyse_luminance.py
   python Code/analyse_vergence.py
   ```

## Data

Participants are identified by pseudonymous IDs (`p000`, `p001`, ...), consistent across the three
experiments. Recordings of one participant in one experiment are numbered in time order
(`grid_1`, `vergence_2`, ...). Timestamps are in seconds on each recording's own clock.

The data come from the limbus model and processing described in the paper. Intervals the analyses
depend on - stimulus presentations, detected eye-movement events, fixations - are given as tables
with exact start and end times, so the scripts reproduce the reported numbers exactly.

### Grid fixation (`Data/1_accuracy/`)

**`fixations.csv`** — one row per fixation period and eye (8,101 rows, 58 participants).

* `participant`, `recording`, `grid` (grid block within the recording), `eye`, `fixation` (target number, in presentation order)
* `t_start_sec`, `t_end_sec`, `n_samples`: the fixation period (700 ms after target onset to 100 ms before offset), and the samples in it
* `accuracy_centroid_deg`: angle between the mean gaze direction and the mean target direction (the accuracy reported in the paper)
* `accuracy_deg`, `accuracy_x_deg`, `accuracy_y_deg`: robust mean of the per-sample angular deviation, and its horizontal and vertical components
* `precision_rms_s2s_deg`: RMS of the sample-to-sample angular differences (the precision reported in the paper); `precision_std_deg`: angular SD about the mean gaze direction
* `stimulus_head_x_deg`, `stimulus_head_y_deg`: horizontal and vertical angle from the limbus centre to the target, in the head (front-facing camera) frame; `stimulus_distance_mm`: distance from the limbus centre to the target
* `head_movement_mm`, `head_movement_deg`: RMS deviation of the head position and orientation from their mean over the fixation
* `stimulus_monitor_x_deg`, `stimulus_monitor_y_deg`: nominal target position on the display grid
* `kappa_x_deg`, `kappa_y_deg`: the eye's kappa (visual-to-optical axis) correction
* `is_fixation_outlier`: 1 if the fixation was rejected as an outlier during kappa estimation
* `participant_in_training_set`: `True` if other recordings of this participant were in the limbus model's training set

**`{participant}_{recording}_{grid}_{eye}.csv`** — per-frame data for the two example recordings
(`p038` for Fig. 4, `p042` for Fig. 5), covering whole fixation periods:

* `timestamp_sec`, `fixation`, `stimulus_monitor_{x|y}_deg`
* `stimulus_dir_{x|y|z}`, `optical_axis_{x|y|z}`, `visual_axis_{x|y|z}`: unit vectors in the head frame (target direction from the limbus centre; measured gaze before and after the kappa correction)
* `optical_deviation_{x|y}_deg`, `visual_deviation_{x|y}_deg`: angular deviation of the gaze from the target
* `in_analysis_window`: the frame lies in the fixation period used for accuracy and precision
* `blink`: the frame was manually marked as a blink and excluded from the accuracy analyses
* `kappa_{x|y}_deg`

### Background luminance change (`Data/2_luminance/`)

**`{participant}_{eye}.csv`** — per-sample gaze and pupil size:

* `timestamp_sec`
* `igX`, `igY`, `igZ`: gaze direction, a unit vector in the head frame; by convention it points *into* the eye
* `azimuth_deg`, `elevation_deg`: the same gaze as angles
* `pupil_diameter_mm`: pupil diameter, with isolated single-sample drops corrected
* `blink`: the sample lies in a detected blink; the analyses treat it as missing

**`protocol.csv`** — the stimulus log: `participant`, `t_start_sec`, `t_end_sec`,
`stimulus_x_deg`, `stimulus_y_deg` (target position), `luminance` (`bright` or `dark` background).
An episode is a dark-background presentation followed by the same target on a bright background.

**`events.csv`** — eye movement events detected by I-VT, per eye: `participant`, `eye`, `event`
(`fixation`, `saccade`, `blink`, `smooth_pursuit`), `t_start_sec`, `t_end_sec`.

### Visual search (`Data/3_vergence/`)

**`{participant}_{recording}_{eye}.csv`** — per-sample eye position and gaze in the marker
coordinate system: the origin is the marker centre, X goes right, Y down, Z away from the
participant; units are millimetres. The two eyes' cameras are not synchronised, so each eye has its
own file and timestamps.

* `timestamp_sec`
* `icX`, `icY`, `icZ`: limbus centre position
* `igX`, `igY`, `igZ`: gaze direction, a unit vector; by convention it points *into* the eye
* `azimuth_deg`, `elevation_deg`: the gaze as angles
* `blink`: the sample lies in a detected blink; the analyses treat it as missing

**`stimuli.csv`** — the six symbols of each run: `participant`, `recording`, `symbol_set`,
`symbol`, `order_from_participant` (1: closest, 6: farthest), `t_start_sec`, `t_end_sec` (when the
symbol was called out).

**`fixations.csv`** — binocular fixations (both eyes fixating, detected by I-VT per eye):
`participant`, `recording`, `t_start_sec`, `t_end_sec`. The reference fixation for each symbol is
derived from these (`analyse_vergence.py`).

## Citation

If you use this data or code, please cite: TBD

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
