## Metrics 

This folder contains everything related to the computed results for this study. 

Now everything is computed inside a class in the **get_all_metrics.py** file, which is computed with the corresponding Shell file. The only thing that needs to be adapted is the Shell file, which has arguments:

- orig_folder: folder to the original test set segmentations (i.e the labelsTs folder).
- preds_folder_name: name of the folder that contains the predicted segmentations.
- preds_folder: the path to preds_folder_name.
- data_info: the path to the Excel file containing all the important data for each pullback (like frames with annots, ID, etc). It basically is the train_test_split_v2 file.
- model_id: arbitrary ID you want to give the model for which you want to get the results. This ID will be used to give the name for all the generated JSON and Excel files.

All the metrics that are obtained are explained below.

### DICE scores

We compute different DICE scores to evaluate the model performance from others points of view:

- Pixel DICE per frame: it is the standard DICE score, computed for each label in the dataset. When we evaluate the test set during inference, a JSON file containing several metrics for each frame in the test set is generated. In order to compute this DICE per frame, the function **dice_per_frame** reads through this JSON file and creates a more organized JSON file with the DICE for each label and each frame. Then, this JSON file is opened with Excel and the results are stored here. 

- Pixel DICE per pullback: it similar to the previous DICE score, but computed per pullback. However, this DICE score is computed directly  in the **dice_per_pullback** function, since this coefficient is not generated in the JSON file. As a results for this code, a different JSON file is created and, again, the results are stored in the Excel file.

- Lipid/calcium arc DICE per frame: this DICE is used to evaluate the performance of the lipid arc automated measurement, which is calculated in the **get_arc_dice_per_frame** function. This DICE is computed by looking at the bins that contain lipid in both manual and automated segmentations. The results are stored in an Excel file.

- Lipid/calcium arc DICE per pullback: similar as the previous case, but the confusion matrix is calculated accross all frames with annotations in a pullback. This is computed in the **get_arc_dice_per_pullback** function. Again, an Excel file is generated.


### Other metrics

Apart from the DICE scores, metrics for detection of every label were obtained, which are obtained with the **get_other_metrics_detection.py**. These metrics are: Positive Predictive Value (PPV), Negative Predictive Value (NPV), sensitivity, specificity and Cohen's kappa. The function **get_other_metrics_pixel.py** computes also these metrics but pixel-wise, rather than computed per detection. 

A JSON file with the average for each label is generated, which is then opened with Excel and stored for each model.







 