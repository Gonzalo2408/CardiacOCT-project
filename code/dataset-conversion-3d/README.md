## 3D dataset conversion

This folder contains the scripts that preprocess the original dataset and return 3D volumes that are later used for the training. There are two specific cases:

### Full scan sampling

For the inital idea for training, we used the full scans and segmentations as input. For that, both images and segmentation scans (full 3D pullbacks) were preprocessed and saved into new pullbacks, keeping the same number of slices. These scripts are saved into the folders "images-conversion" and "segs-conversion". In this case, there is a Docker file for both images and segmentations, thanks to which this conversion can be run in SOL.

Using this data as input for the training leads to several problems:

 - **NaN loss**: since the annotations are very sparse in compare with the size of the pullback, the sparse trainer cannot calculate a loss function (difficult to troubleshoot this, but I suppose it's because of gradient exploding or zero division error while calculating the loss and DICEs)

 - **Long runtimes**: for the full dataset, one epoch took around 3000 seconds (~1 hour). This makes this training not feasible or reproducible, considering the fact that the algorithm runs for 1000 epochs.


![Figure 1. Preprocessing framework for the 3D volumes](/assets/3d_dataset_conversion.png)

### Selective scan sampling

In this case, we only create volumes sampling around annotations instead of using the full pullback as input for the model. The scripts for this are in the pullback_splits folder. We are trying two different approaches:

 - **Sample n frames around annotation**: for each annotation, we use the frames before and after and the annotation itself to build the new 3D volumes. In this case, we sample one frame before and one after, creating a volume with 3 slices. However, the 3D training fails since this volumes are very thin, which causes a crashing in the data augmentation steps (probably due to that there is a minimum of axial slices).


 - **Cluster neighboring annotations**: in this case, following a "clustering" idea, we obtain the neighbor annotations for each annotation. Using this annotation, we generate a volume that contains all of the slices in between these annotations, plus one frame before the first annotation and one frame after the last annotation. With this, we hope to solve the previous problems.


 ### Additional scripts

 Similarly to the 2D conversion, file_and_folder_operations.py and generate_json_dataset.py are used to generate the .json file needed for the training of the data.

 Moreover, the add_minus_ones.py insert -1 in slices that do not have an associated label. This script, however, was used for testing purposes (the -1s are already being insert while generating the 3D segmentations)

