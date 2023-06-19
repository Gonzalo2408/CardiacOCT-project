## Metrics and results analysis

This folder contains everything related to the computed results for this study. 

### DICE scores

The nnUNet returns many different metrics for each image we have in our dataset (for train/val/test sets) and it stores them into .json files. Using the scripts in **get_dice_frame_level.py** and **get_dice_pullback_level.py** we could take the DICE scores from those .json files. The DICE scores are obtained:

 - Frame-wise: the confusion matrix is computed for each frame independently.
 - Pullback-wise: the confusion matrix is computed for a whole pullback, that is, for every frame with annotation belonging to the same pullback. With this approach, we avoid having more NaN DICEs in our results, getting a more accurate DICE

 These DICEs are computed by seeing each pixel. However, for the lipid and calcium, we are also interested in their respectives arc. That is why we computed the DICE scores for the lipid arc, as in [Lee et al](https://www.nature.com/articles/s41598-022-24884-1). The way to do this is to look into the post-processing measurements. For each bin that is generated in a frame (there are 360 bins), each one can either contain lipid/calcium or not. By using this, we can find the confusion matrix for both predicted segmentation and manual segmentation (we basically see which bins occur in one case and the other). This way we can get a better estimate on how good or bad are the lipid and calcium segmentations, adding more clinical value to the segmentation of these. Similarly as with the other DICE scores, we provide the lipid and calcium arc DICE both frame and pullback-wise. The script that performs this task is **get_angle_dices.py**.

 The results can be seen in the Excel files in the [metrics](info-files) folder in this repository


### Statistical analysis

The statistical analyis on lipid (arc and FCT) and calcium (arc, thickness and depth) is also performed in this folder, with the aim to compare manual and automated measurements on these regions. Essentially, we generate an Excel file with the following information:

 - False Positive and False Negatives
 - Bland-Altman plots: plus mean difference and standard deviation for manual and automated measurements
 - Intra-class correlation (ICC)
 - Correlation + plot
 - Outliers: using either Z-score or Tukey approaches.

 The Excel file and the results with this can be seen in the [statistics](info-files/statistics) folder





 