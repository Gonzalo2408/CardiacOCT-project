## Distributions

All the code that calculates the class weights and distributions for the different datasets are found in this folder. Now, everything is compiled in one class in the **get_distributions.py** file, which is compiled using the corresponding Shell file. The Python class compiles:

 - Get counts: it reads every image in the train or test set and creates an Excel file with these counts. For each frame, it gives either 0 or 1 if the label is present or not, and also it perfoms the automated measurements. In addition, the lipid and calcium sizes are also stored for later thresholding.

 - Get class weights: it returns the class weights for the train set, which was used for some of the trainings. The way the weights are calculated is by obtaining the ratio of the amount of pixels with a specific label to the total ammount of pixels in the training set.

 In order to reproduce this code, you should specify the function you want to run in the Python script (basically, in the case if you want to get the class weights) and adapt the paths of the Shell file. The arguments in this file are explained:

 - data_path: the path to the folder containing either the model predictions, or the raw segmentations in the case you want to count the distributions on the manual segmentations (i.e would need to be the labelsTs). You can run it for the training set (labelsTr if you want the manual segs, or the cv_niftis_postprocessed for the training results), but it takes too long and it is not really necessarily.

 - output_filename: the path to the Excel file you want to generate (you should specify also the name of the Excel file in the path, without the extension).

 - data-info: the path to the Excel file containing all the important data for each pullback (like frames with annots, ID, etc). It basically is the train_test_split_v2 file.


 ## Lipid and calcium size thresholds

 The notebook **get_lipid_thresh.py** obtains the ROC curves thanks to which an optimal threshold for detected lipid and calcium size is found. These sizes are obtained with the distributions file as well, and stored in an Excel file. In this notebook, by using this Excel, we obtain the optimal threshold by analyzing the sizes in the train results on the pseudo 3D with k = 3 model. An optimal value of 1700 and 100 pixels is found for lipid and calcium regions, respectively. Finally, the threshold is applied to the test set predictions and the sensitivity and specificity are again computed.
