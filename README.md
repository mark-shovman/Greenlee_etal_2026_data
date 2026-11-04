# Accurate AI-assisted binocular limbus eye tracking in participants performing visually guided eye-movement tasks

This repository contains the data and analysis code for the paper:

> Greenlee M, Keil E, Plank T, Jägle H, Lee Chen D M, Lu I, Vorobey A, Shovman M, Kashchenevsky A.
>
> **Accurate AI-assisted binocular limbus eye tracking in participants performing visually guided eye-movement tasks.**

The paper is currently in review, and the details will be updated here if and when it is published.

The scripts in `Code/` reproduce the paper's Figures 7–18, Table 1, and the statistics quoted in
the Results from the data in `Data/`.

## Repository Structure

```text
.
├── Code/                   # Analysis and plotting scripts
│   ├── analyse_accuracy.py    # Grid fixation: Figs 7-12, Table 1
│   ├── analyse_luminance.py   # Background luminance change: Figs 13-15
│   ├── analyse_vergence.py    # Visual search: Figs 16-18
│   ├── participant_table.py   # Per-participant supplementary table
│   └── utils.py
├── Data/
│   ├── 1_accuracy/
│   ├── 2_luminance/
│   ├── 3_vergence/
│   └── participants.csv    # Age group, sex, refraction
├── Tables/                 # Per-participant table (participants.*), kappa per grid block (kappa_blocks.*)
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

## Per-participant table

`Tables/participants.csv` (full precision) and `Tables/participants.md` (rounded) give one row per
participant; `python Code/participant_table.py` regenerates both. Cells are empty where a
participant did not take part in that experiment.

* `in_training_set`: other recordings of this participant were in the limbus model's training set
* `age_group` (decade), `sex`, `refraction_{left|right}_D`: from `Data/participants.csv`; refraction is empty where it was not recorded
* `kappa_{left|right}_{x|y}_deg`: kappa correction per eye, median over the participant's grid blocks (`Tables/kappa_blocks.*`)
* `n_fixations`, `n_fixations_valid`: all fixations, and the valid ones (`is_fixation_valid`)
* `{accuracy|precision}_median_deg`: unweighted median over the valid fixations
* `{accuracy|precision}_{whole|central}_deg`: inverse-density-weighted median over the whole FOV or the central 15×15°, i.e. the participant's E50 in Figs 11–12; `_ci_low`, `_ci_high`: bootstrap 95% CI (2,000 resamples of the participant's fixations, each keeping its weight); `n_{accuracy|precision}_{whole|central}`: fixations in that weighting
* `lum_valid_episode_eyes`: luminance episodes × eyes without a saccade or blink between −1,300 and 1,500 ms in either eye, and with gaze and pupil in at least 30% of the expected samples of each analysis window in both eyes (of 6); `lum_constriction_mm`, `lum_gaze_shift_deg`: their mean pupil constriction and apparent gaze shift (baseline −500–0 ms, recovery 1,000–1,500 ms)
* `vs_runs`, `vs_stimuli_valid`: visual search runs, and symbols with a reference fixation
* `vs_decoding_{within|cross}`: leave-one-out depth decoding accuracy (Fig. 18)
* `vs_pairs_discriminated`, `vs_pairs_testable`: adjacent depth pairs discriminated at p_adj < 0.001, of those testable

`Tables/kappa_blocks.csv` / `.md`, written by the same script, give kappa per participant, eye, recording and
grid block (`kappa_grid_{x|y}_deg` in `fixations.csv`), with the block's number of valid fixations.

## Data

Participants are identified by pseudonymous IDs (`p000`, `p001`, ...), consistent across the three
experiments. Recordings of one participant in one experiment are numbered in time order
(`grid_1`, `vergence_2`, ...). Timestamps are in seconds on each recording's own clock.

The data come from the limbus model and processing described in the paper. Intervals the analyses
depend on - stimulus presentations, detected eye-movement events, fixations - are given as tables
with exact start and end times, so the scripts reproduce the reported numbers exactly.

### Participants (`Data/participants.csv`)

One row per participant (58): `participant`, `age_group` (decade, e.g. `20-29`), `sex` (`F`/`M`),
`refraction_left_D`, `refraction_right_D` (refractive correction in dioptres). Refraction was
recorded for 18 participants and is empty for the others.

### Grid fixation (`Data/1_accuracy/`)

**`fixations.csv`** — one row per fixation period and eye (7,978 rows, 58 participants; 7,976 valid). One grid
block is left out: `grid2` of `p038`'s `grid_2` recording, where the stimulus presentation system
stalled repeatedly and skipped 14 of the 61 targets.

* `participant`, `recording`, `grid` (grid block within the recording), `eye`, `fixation` (target number, in presentation order)
* `t_start_sec`, `t_end_sec`, `n_samples`: the fixation period (700 ms after target onset to 100 ms before offset), and the samples in it
* `accuracy_centroid_deg`: angle between the mean gaze direction and the mean target direction (the accuracy reported in the paper)
* `accuracy_deg`, `accuracy_x_deg`, `accuracy_y_deg`: robust mean of the per-sample angular deviation, and its horizontal and vertical components
* `precision_rms_s2s_deg`: RMS of the sample-to-sample angular differences between consecutive samples, at most 12.5 ms (1.5 frames) apart, so a gap left by removed frames is not counted as a step (the precision reported in the paper); `precision_std_deg`: angular SD about the mean gaze direction (the SD row of Table 1)
* `stimulus_head_x_deg`, `stimulus_head_y_deg`: horizontal and vertical angle from the limbus centre to the target, in the head (front-facing camera) frame; `stimulus_distance_mm`: distance from the limbus centre to the target
* `head_movement_mm`, `head_movement_deg`: RMS deviation of the head position and orientation from their mean over the fixation
* `stimulus_monitor_x_deg`, `stimulus_monitor_y_deg`: nominal target position on the display grid
* `kappa_x_deg`, `kappa_y_deg`: the kappa (visual-to-optical axis) correction applied to this fixation. The grid's targets are split into two halves in a checkerboard pattern (`kappa_half`, 0 or 1), and each half is corrected with the kappa fitted on the other half, so kappa is never fitted on the fixations it is scored on. Kappa is the median, over the other half's valid fixations, of each fixation's mean horizontal and vertical gaze deviation from the target
* `kappa_grid_x_deg`, `kappa_grid_y_deg`: the same median over all valid fixations of the grid block; reported, not applied
* `frac_tracked`, `frac_head`, `frac_unannotated`: share of the expected samples (fixation period × 120 Hz eye camera, × 30 Hz for head pose) that the tracker delivered, that had a head pose, and that were left after removing the frames annotated as blinks
* `is_fixation_valid`: 1 if all three shares are at least 0.3. The analyses use the valid fixations; no fixation is rejected for its error
* `participant_in_training_set`: `True` if other recordings of this participant were in the limbus model's training set

**`{participant}_{recording}_{grid}_{eye}.csv`** — per-frame data for the two example recordings
(`p038` for Fig. 7, `p042` for Fig. 8), covering whole fixation periods:

* `timestamp_sec`, `fixation`, `stimulus_monitor_{x|y}_deg`
* `stimulus_dir_{x|y|z}`, `optical_axis_{x|y|z}`, `visual_axis_{x|y|z}`: unit vectors in the head frame (target direction from the limbus centre; measured gaze before and after the kappa correction)
* `optical_deviation_{x|y}_deg`, `visual_deviation_{x|y}_deg`: angular deviation of the gaze from the target
* `in_analysis_window`: the frame lies in the fixation period used for accuracy and precision
* `blink`: the frame was manually marked as a blink and excluded from the accuracy analyses
* `kappa_{x|y}_deg`: the kappa applied to the frame's fixation (from the other checkerboard half)

### Background luminance change (`Data/2_luminance/`)

**`{participant}_{eye}.csv`** — per-sample gaze, target direction and pupil size:

* `timestamp_sec`
* `igX`, `igY`, `igZ`: gaze direction, a unit vector in the head frame; by convention it points *into* the eye
* `azimuth_deg`, `elevation_deg`: the same gaze as angles
* `pupil_diameter_mm`: pupil diameter, with isolated single-sample drops corrected
* `stimulus_dir_x`, `stimulus_dir_y`, `stimulus_dir_z`: direction between the limbus centre and the target, a unit vector in the head frame, pointing *into* the eye like `igX/igY/igZ`. The displayed target position is carried into the head frame by the head pose, so it follows head movement; empty where there is no head pose
* `stimulus_azimuth_deg`, `stimulus_elevation_deg`: the same direction as angles, comparable with `azimuth_deg`, `elevation_deg`
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
