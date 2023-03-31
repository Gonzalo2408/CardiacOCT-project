## Metrics and results analysis

This folder contains everything related to the computed results for this study. 

### DICE scores

The nnUNet returns many different metrics for each image we have in our dataset (for train/val/test sets) and it stores them into .json files. Using the scripts in **build_results.py** and **merge_dices.py** we could take the DICE scores from those .json files.

 The **build_results.py** script just returns the DICE scores per pullback by averaging over the total number of frames in that pullback. The DICEs were the ones returned by the nnUNet. The .json files in **model_1.py**, **model_2.py** and **model_3** contain these DICE results. On the other hand, the **merge_dices.py** script returns the DICEs pullback level. That is, instead of computing the confusion matrix per frame, we compute the confusion matrix per pullback, so that way we obtain a more exact DICE result since we can avoid having more NaN DICEs in our results. For this, we needed to run through each nnUNet segmentation and compute this new DICE scores. These DICEs can be seen in the **pullback-wise dices**.

These DICEs are computed by seeing each pixel. However, for the lipid, we are also interested in the lipid arc. That is why we computed the DICE scores for the lipid arc, as in [Lee et al](https://www.nature.com/articles/s41598-022-24884-1). The way to do this is to look into the post-processing measurements. For each bin that is generated in a frame (there are 360 bins), each one can either contain lipid or not. By using this, we can find the confusion matrix for both predicted segmentation and manual segmentation (we basically see which bins occur in one case and the other). This way we can get a better estimate on how good or bad are the lipid segmentations, adding more clinical value to this lipid segmentation. Similarly as with the other DICE scores, we provide the lipid arc DICE both frame and pullback-wise. The script that performs this task is **lipid_angle_dices.py**.

All of the DICE results (frame-level, pullback-level and lipid arc) can be seen in the **metrics_models.xlsx**.

### Comparisons

This folder contains a comparison of the Silvan's initial model and the latest model (model 3 2D). The results can be seen in the Excel file in that folder, which just contains the DICEs scores for the latest model and Silvan's model. IMPORTANT: note that we can only do this comparisons using frames that are in both Silvan's test set and my test set. This is why we could only compare the frames in one pullback (NLD-AMPH-0054).


### Bland-altman plots

The Bland-Altman script can be seen as well in this folder. It basically uses the lipid arc and FCT measurements from the predictions and the manual lipid arc and FCT measurements from the original segmentations as input (note that FP and FN are deleted for this analysis, since this can introduce some bias).





 