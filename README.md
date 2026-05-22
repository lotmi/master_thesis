# Repository for the Master Thesis project by Lotte Michels

This repository contains the code that was used for the reported deep learning experiments.

## Code Tree: 
 * [analysis_and_visualization](./analysis_and_visualization): notebooks for data exploration and visualization of the experimental results. 
 * [CODE](./CODE): all code scripts to run the deep learning experiments.
   * [TotalSegmentator](./CODE/TotalSegmentator): the scripts used to process the TotalSegmentator scans and develop models for the bone detection task.
     * [preprocessing_modules](./CODE/TotalSegmentator/preprocessing_modules): functions for data pre-processing.
     * [YOLO](./CODE/TotalSegmentator/YOLO): YOLO development code.
     * [RetinaNet](./CODE/TotalSegmentator/RetinaNet): RetinaNet development code.
   * [ETZ](./CODE/ETZ): the scripts used to process the Kahler dataset and develop models for the osteolytic lesion detection task.
     * [preprocessing_modules](./CODE/ETZ/preprocessing_modules): functions for data pre-processing.
     * [YOLO](./CODE/ETZ/YOLO): YOLO development code.
     * [RetinaNet](./CODE/ETZ/RetinaNet): RetinaNet development code.
     * [sahi_adjustments](./CODE/ETZ/sahi_adjustments): two scripts from the SAHI source code (Akyon et al., 2021) that were adapted for this thesis project.
 * [requirements.txt](requirements.txt): the libraries and library versions used for this thesis project.

## Pre-processing Code:
Code for data pre-processing was partly re-used from my internship project, which is described in the working paper by Michels et al. (2026). 

## SAHI Source Code Adjustments:
Code added to the SAHI source code (Akyon et al., 2021) was done in the following two scripts:
* `predit.py` in lines 2020203 and lines 321-322
* `slicing.py` in lines 296-300 and lines 361-383
The adapted scripts are included in: [sahi_adjustments](./CODE/ETZ/sahi_adjustments).
Adjustments are clearly indicated with comments on and around the mentioned code lines. 

## Reference:
Akyon, F. C., Cengiz, C., Altinuc, S. O., Cavusoglu, D., Sahin, K., & Eryuksel, O. (2021). SAHI: A lightweight vision library for performing large scale object detection and instance segmentation. Zenodo. Retrieved from https://doi.org/10.5281/zenodo.5718950

Michels, L., van Leeuwen, M., & Ong, L. L. S. (2026). Decreasing the False Positive Rate in YOLO Lung Nodule Detectors with Negative Sampling. [Working paper]


  

