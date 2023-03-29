## Labels predictions

In this folder, you can find the scripts that use the predicted labels from nnUNet for further analysis. This information is stored in the following Excel files:

- **counts.xlsx**: the overall results for each training are stored here. Different sheets can be found:

    - Plots: you can see here an overview of how common is each label in proportion with the number of annotated frames. You can see how the different train models represent this data distribution
    - Overview dataset / model: the full representation for each dataset and training. This ditribution is checked among all of the annotated frames and all pullbacks.
    - Pullback counts X: this just gives the count for each labels among a single pullback, being X either the dataset or one of the models used for training.
    - Frames counts X: same as with the pullback, but for each annotated frame. In this case, each cell gives a binary value indicating that if a label is present or not in the current frame. Moreover, for the cases in which X corresponds to a trained model, you can also see the post-processing measurements for that frame (lipid arc and cap thickness), plus if a TCFA is detected or not.

- **manual_vs_automatic_fct_arc.xlsx**: the goal of this Excel file is to see more in detail what is the difference between the manual and automatic measurements of the FCT and lipid arc for the same manual segmentations in the test set. In theory, both measurements should be the same (since they are seeing the same manual segmentation), but there is some bias in different measurements. We are currently analyzing each case to see why is this bias happening.


The two scripts generate these Excel files (one script for counting in the original datasets and other one for the predicted results). Moreover, for the test images, the segmentation and automatic measurement are saved as .pngs into a folder for further visual analysis. This is also done for both manual and predicted segmentations.