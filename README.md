# Accurate AI-assisted binocular limbus eyetracking in participants performing visually guided eye-movement tasks

[//]: # ([![DOI]&#40;https://img.shields.io/badge/DOI-10.3758/s13428--025--02880--3-blue&#41;]&#40;https://doi.org/10.3758/s13428-025-02880-3&#41;)
[//]: # ([![Paper]&#40;https://img.shields.io/badge/Paper-Behavior%20Research%20Methods-green&#41;]&#40;https://rdcu.be/e49pe&#41;)

This repository contains the data and analysis code for the paper:

> Greenlee M, Keil E, Plank T, Jägle H, Lee Chen D M, Lu I, Vorobey A, Shovman M, Kashchenevsky A.
> Accurate AI-assisted binocular limbus eyetracking in participants performing visually guided eye-movement tasks.
 
The paper is currently in review, and the details will be updated here if and when it is published. 


## Repository Structure
The repository is structured as follows:
* `Data/` contains the raw eye-tracking data from the experiments  
* `Code/` contains the statistical analysis and plotting code in Python

## Data

### Accuracy Experiment Data (`Data/1_accuracy/`)   

Contains eye-tracking data for the first experiment; file names follow the format `{participant_id}_{trial_id}_{eye}.csv`.

The columns of the data files are:
* `timestamp_sec`: Timestamp in seconds.
* `deviation_from_stimulus_{x|y}_dva`: Deviation of the eye position from the stimulus position in degrees of visual angle (dva).
* `stimulus_{x|y}_deg`: Stimulus position in degrees, relative to the monitor center.
* `kappa_{x|y}_deg`: visual-to-optical axis correction in degrees – constant for each eye.
* `is_fixation_outlier`: True if this fixation was considered an outlier – constant for this stimulus coordinate.
* `participant_in_training_set`: True if this participant was in the training set – constant for this participant.

### Luminance Experiment Data (`Data/2_luminance/`)   

Contains eye-tracking data for the second experiment; file names follow the format `{participant_id}_{eye}.csv`.

* `timestamp_sec`: Timestamp in seconds.
* `{x|y}_dva`: Gaze relative to the head (Fick's angles).
* `pupil_diameter_mm`: Pupil diameter in millimeters.
* `stimulus_{x|y}_deg`: Stimulus position in degrees, relative to the monitor center.
* `luminance`: monitor luminance level, either 'bright' or 'dark'.
* `fixation_id`: Unique identifier for each fixation detected by I-VT.

## Code


## Citation
TBD

[//]: # (If you use this data or code, please cite: )

[//]: # ([![DOI]&#40;https://img.shields.io/badge/DOI-10.3758/s13428--025--02880--3-blue&#41;]&#40;https://doi.org/10.3758/s13428-025-02880-3&#41;)

## License

See [LICENSE](LICENSE) for details.