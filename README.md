# Accurate AI-assisted binocular limbus eyetracking in participants performing visually guided eye-movement tasks

[//]: # ([![DOI]&#40;https://img.shields.io/badge/DOI-10.3758/s13428--025--02880--3-blue&#41;]&#40;https://doi.org/10.3758/s13428-025-02880-3&#41;)
[//]: # ([![Paper]&#40;https://img.shields.io/badge/Paper-Behavior%20Research%20Methods-green&#41;]&#40;https://rdcu.be/e49pe&#41;)

This repository contains the data and analysis code for the paper:

> Greenlee M, Keil E, Plank T, Jägle H, Lee Chen D M, Lu I, Vorobey A, Shovman M, Kashchenevsky A.
> 
> **Accurate AI-assisted binocular limbus eyetracking in participants performing visually guided eye-movement tasks.**

The paper is currently in review, and the details will be updated here if and when it is published. 

## Repository Structure

```text
.
├── Code/                   # Statistical analysis and plotting scripts
│   ├── analyse_accuracy.py
│   ├── analyse_luminance.py
│   ├── analyse_vergence.py
│   └── utils.py
├── Data/                   # Raw eye-tracking data
│   ├── 1_accuracy/
│   ├── 2_luminance/
│   └── 3_vergence/
├── LICENSE
├── README.md
└── requirements.txt        # Python dependencies
```

## Data

### Accuracy Experiment Data (`Data/1_accuracy/`)

Contains eye-tracking data for the first experiment. File names follow the format: `{participant_id}_{trial_id}_{eye}.csv`.

**Columns:**
* `timestamp_sec`: Timestamp in seconds.
* `deviation_from_stimulus_{x|y}_dva`: Deviation of the eye position from the stimulus position in degrees of visual angle (dva).
* `stimulus_{x|y}_deg`: Stimulus position in degrees, relative to the monitor center.
* `kappa_{x|y}_deg`: Visual-to-optical axis correction in degrees (constant for each eye).
* `is_fixation_outlier`: `True` if this fixation was considered an outlier.
* `participant_in_training_set`: `True` if this participant was in the training set.

### Luminance Experiment Data (`Data/2_luminance/`)

Contains eye-tracking data for the second experiment. File names follow the format: `{participant_id}_{eye}.csv`.

**Columns:**
* `timestamp_sec`: Timestamp in seconds.
* `{x|y}_dva`: Gaze relative to the head (Fick's angles).
* `pupil_diameter_mm`: Pupil diameter in millimeters.
* `stimulus_{x|y}_deg`: Stimulus position in degrees, relative to the monitor center.
* `luminance`: Monitor luminance level (`bright` or `dark`).
* `fixation_id`: Unique identifier for each fixation detected by I-VT.

### Vergence Experiment Data (`Data/3_vergence/`)

Contains eye-tracking data for the third experiment. File names follow the format: `{participant_id}_{trial_id}.csv`.

The data is for both eyes, in the marker coordinate system: the origin is the marker center, X goes right, Y down, Z away from the participant. Units are in millimeters.

**Columns:**
* `timestamp_sec`: Timestamp in seconds.
* `ic{X|Y|Z}_{left|right}`: Limbus center position, in millimeters, relative to the marker center.
* `ig{X|Y|Z}_{left|right}`: Gaze vector relative to the marker center. By convention, the gaze vector points _into_ the eye.
* `stimulus_order_from_viewers`: Stimulus order from the participant's perspective (1: closest, 6: farthest).
* `reference_fixation_id`: If not empty, this row is part of the fixation used as a reference period for the vergence measurement.

## Code and Analysis

The `Code/` directory contains Python scripts for analyzing the eye-tracking data:

* `analyse_accuracy.py`: Analyzes data from the accuracy experiment.
* `analyse_luminance.py`: Analyzes data from the luminance experiment.
* `analyse_vergence.py`: Analyzes data from the vergence experiment.
* `utils.py`: Common utility functions.

### Getting Started

1. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis:**
   Execute the scripts from the root directory:
   ```bash
   python Code/analyse_accuracy.py
   python Code/analyse_luminance.py
   python Code/analyse_vergence.py
   ```

## Citation


If you use this data or code, please cite: TBD

[//]: # ([![DOI]&#40;https://img.shields.io/badge/DOI-10.3758/s13428--025--02880--3-blue&#41;]&#40;https://doi.org/10.3758/s13428-025-02880-3&#41;)

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.