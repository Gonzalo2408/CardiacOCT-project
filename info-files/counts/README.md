## Distributions file

The file in this folder contains everything related to getting the distributions of the manually segmented dataset and the different model predctions. Below, a short explanation of every sheet in this file:

### Plots

This sheet shows, up to now, two plots: one for a comparison of the test set distribution for every model and the manual test set, and another plots for a comparison of the distribtution of the labels across the train and test sets. The aim of these plots is to easily visualize if there are so biases towards underrepresenting or overrepresenting certain labels. 

Figure 1 shows a similar distribution across every model and test set. It is worth pointing out, however, an overrepresentation in lipid for every model (from a bit less than 50% to 60%).

![Figure 1. Test set distribution comparison accross every model](/assets/models_dists.png)

Figure 2 shows the train/test distribution. Except for some underrepresentation in calcium and overrepresentation on sidebranch, the classes are similarly distributed across both sets. Notwithstanding, all the frames containing dissection belong to the train set, meaning that no dissections are found in the test set.

![Figure 2. Raw train/test set distribution](/assets/train_test_dists.png)

### Overview

Here, several tables give a summary of the exact percentages of the labels distributions. The figures above explained are basically obtained using the values in this sheet. For every model and train and test sets, we obtain:

 - Nº of frames containing a label
 - Nº of pullbacks containing a label
 - % of frames compared to the total nº of frames
 - % of pullbacks compared to the total nº of pullbacks


### Frames sheets

Every model (including the train and test sets) has a sheet containing the label values for each frame, with the automated measurements for that frame. Each frame can either have a value of 0 or 1 for every label. For the measurements, if there is no lipid/calcium, the measurement will give -99. Note that for the models, we only include the results on the test set, not the training set.

### Pullbacks sheets

Similary, each model and train and test sets has a sheet with the pullback distributions. Every row corresponds to a pullback, and the total nº of frames containing every label. Again, we only obtain these results for the test set.
