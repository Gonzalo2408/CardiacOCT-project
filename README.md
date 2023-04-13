# CardiacOCT project

## Project description

Acute myocardial infarction (MI) remains as one of the leading causes of mortality in the developed world. Despite huge advances in treating this condition such as the use of angiograms to locate the occluded artery or coronary angioplasty, there is still a debate on whether to treat certain lesions found during coronary angiography or not, since it is difficult to predict which plaques would have a worse outcome for the patient’s health. Imaging modalities such as intracoronary optical coherence tomography (OCT) provide a better comprehension of plaque characteristics and it can help surgeons to better asses these lesions, improving patient’s outcome.

In this project, an automatic segmentation model will be designed for intracoronary OCT scans in order to asses for plaque vulnerability and detect other abnormalities such as white or red thrombus or plaque rupture. Specifically, a no-new UNet (nnUNet) that works with sparse annotated data will be designed. Initially, the model will be trained on singles frames that contain a corresponding segmentation map, that is, the model works in a supervised manner. Next, in order to account for the sparse annotations, a 3D UNet will be trained in a semi-supervised manner. After the models have been trained, several automatic post-processing techninques for lipid arc and cap thickness measurement will be implemented. Moreover, an uncertainty estimation model will be designed in order to detect unreliable segmentations and add more value to the algorithm's output.

![Figure 1. Example of intracoronary OCT frame (left) with its corresponding manual segmentation (right)](assets/intro_images.png)
           
## Dataset

The intracoronary OCT dataset used in this study is a collection of OCT scans from 5 different medical centers: Isala (ISALA, Zwolle), Amphia Hospital (AMPH, Breda), North Estonia Medical Center (NEMC, Tallinn), Den Haag Medical Centrum (HMC, Den Haag) and RadboudUMC (RADB, Nijmegen).

Since the manually labelling of OCT frames is a very time consuming task for annotators, not all scans were included for the training. In particular, the standard methodology is to label each 40th frame in the scan, unless there are some regions in other frames that are necessarily to label. Thus, frames that contain some degree of labelling were included for the training. 

| Dataset  | Nº of patients (train/test) | Nº of pullbacks (train/test) | Nº of annotated frames (train/test)
| ------------- | ------------- | -------------  | -------------
| First dataset  | 49/13 (1 EST-NEMC, 24 AMPH, 3 HMC, 24 ISALA, 10 RADB)  | 56/14  | 783/163
| Second dataset  | 75/13 (1 EST-NEMC, 27 AMPH, 3 HMC, 24 ISALA, 33 RADB)  | 88/14  | 1215/162
| Third dataset  | 100/13 (1 EST-NEMC, 33 AMPH, 3 HMC, 24 ISALA, 52 RADB)  | 118/14  | 1649/162 
| Fourth dataset  | 112/13 (1 EST-NEMC, 33 AMPH, 3 HMC, 24 ISALA, 64 RADB)  | 134/14  | 1846/162 


We show the regions of interest (ROIs) that the algorithm segments. With the aim to understand better the dataset, we obtained the distribution for each ROI among the three datasets that were used in the study. These values can be seen in the following table. 

| ROI  | First dataset (frames/pullbacks)(%) | Second dataset (frames/pullbacks) (%) | Third dataset (frames/pullbacks)(%) | Fourth dataset (frames/pullbacks)(%) | Test set (frames/pullbacks)(%)
| ------------- | ------------- | ------------- | ------------- | -------------
| Lumen  | - | - | - | - | -
| Guidewire  | - | - | - | - | -
| Wall | - | - | - | - | -
| Lipid | 51.08 / 98.21 | 46.74 / 97.72 | 47.96 / 96.61 | 48.26 / 97.01 | 48.14 / 92.85 
| Calcium | 27.58 / 83.92 | 27.07 / 81.81 | 31.59 / 83.05 | 32.12 / 84.32 | 16.66 / 71.42
| Media | 94.89 / 100 | 96.21 / 100 | 94.9 / 100 | 95.07 / 100 | 99.38 / 100
| Catheter | - | - | - | - | - | -
| Sidebranch | 13.79 / 85.71 | 14.97 / 89.77 | 15.46 / 89.83 | 15.81 / 88.81 | 16.67 / 71.42 
| Red thrombus | 6.89 / 26.78 | 5.67 / 23.86 | 6.67 / 24.57 | 6.07 / 23.13 | 0.61 / 7.14
| White thrombus | 5.61 / 28.57 | 4.53 / 23.86 | 5.45 / 27.96 | 4.87 / 24.62 | 0 / 0
| Dissection | 0.76 / 5.35 | 0.49 / 3.41 | 0.36 / 2.54 | 0.32 / 2.23 | 0 / 0
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 7.09 / 20.33 | 6.44 / 19.40 | 3.08 / 14.28


Note that the lumen, guidewire, wall and catheter are present in every frame of the datatset.


## Preprocessing

The general preprocessing consisted of reshaping the images to a common size, which was (704, 704) and applying a circular mask to each slice. This is because each slice contains a watermark by Abbott with a small scale bar, and we do not want our algorithm to learn from this information.

### 2D approach

For the 2D approach, the slices that did not contain any label were omitted. Thus, each slice for every pullback in the dataset was saved to a single NifTI file. In addition, each channel in the slice (RGB values) were saved separately as well, obtaining 3 files for each frame in the pullback. Similiary, each segmentation frame was saved in a different NifTI file. In this case, the segmentation is 1-dimensional, so there was no need to create a file for each dimension.

For the first and second training, a linear interpolation resampler was used for both segmentations and images. In the case of the images, a circular mask with radius 340 was applied. Next, each 2D frame was converted to a pseudo 3D scan by including and extra dimension of shape 1, having a final shape of (1, 704, 704). Finally, the spacing and direction of the frame was set to (1.0, 1.0, 1.0) and (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), respectively.

For the third training, a nearest neighbor interpolation sampler was used, since the liner interpolator introduces more artifacts in the frame. After this step, due to missing segmentations on the edge of the frame, a circular mask with radius 346 was applied to both images and segmentation, so the overlap between them is now perfect. Again, the frames were converted to pseudo 3d scans. However, the spacing of the frame was changed to (1.0, 1.0, 999.0), to avoid possible conflicts with calculating the transpose of the image.

### 3D approach

For the 3D version of the nnUNet, a sparse trainer was used. In this case, the loss function is computed using slices that contain annotations in each 3D volume (DC + CE loss). The frames that do not contain any label have a segmentation map that only contains -1, in order for the algorithm to detect unlabeled data. The preprocessing steps are very similar to the 2D model (third training), in which each pullback is separated into its RGB values and each volume is saved separately in different NifTI files. Then, the main difference is that now whole 3D volumes are saved, rather than single 2d frames.


## Training

### nn-UNet

The no-new UNet (nnUNet) is based on the well-known UNet architecture. In this model, an encoder part downsizes the input and increases the number of feature channels, followed by a decoder that takes upsamples the feature maps and reconstructs the original size of the input image. These enconder and decoder networks are also connected by skip connections that allow the decoder to use high-resolution features from the encoder.

The problem with this model is that it needs a very specific input settings and preprocessing the data can be a tedious task. That is why not only the nnUNet uses the U-Net architecture, but also it automatically configures itself, including the preprocessing, network architecture, training and post-processing for any task and data. Hence, while achieving state-of-the-art performances in different tasks, nnUNet adds a systematic framework that can overcome problems and limitations during manual configurations. 

For more information on the nnUNet architecture and processes, see the [original repository](https://github.com/MIC-DKFZ/nnUNet), the [original paper](https://www.nature.com/articles/s41592-020-01008-z) and the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf).

## Post-processing

For the post-processing techniques, an algorithm that automatically measures the fibrous cap thickness (FCT) and the lipid arc was developed. A plaque is usually deemed vulnerable when a thin-cap fibroatheroma (TCFA) appears or there are either cap rupture or thrombus formation. In the case of TCFA, this occurs when there is a lipd arc ≥ 90º and a FCT < 65 µm. That is why the correct measurement of these two values is very important for the correct treatment of the patient.

![Figure 2. Example of nnUNet prediction (left) with the measured lipid arc and cap thickness (right)](assets/post_proc_intro_image.png)

## Results

We obtained several metrics (accuracy, recall, jaccard, etc), but we only diplay the DICE scores for each one of the regions segmented. For the 2D models, the DICE scores computed per frame are showed. In order to accurately compare the 2D models and the 3D model, the DICE scores for the 2D models were computed pullback-wise (i.e the values of the confusion matrix are computed using every pixel in the pullback, rather than for every frame independently). 

### Results of best cross-validation model

| ROI  | 2D model 1st dataset | 2D model 2nd dataset | 2D model 3rd dataset | 3D sparse model
| ------------- | -------------- | -------------- | -------------- | -------------- 
| Lumen  | 0.975 | 0.979 | 0.987
| Guidewire  | 0.917 | 0.923 | 0.946
| Wall | 0.879 | 0.89 | 0.899
| Lipid | 0.467 | 0.485 | 0.519
| Calcium | 0.389 | 0.42 | 0.51
| Media | 0.74 | 0.758 | 0.767
| Catheter | 0.979 | 0.982 | 0.992
| Sidebranch | 0.414 | 0.449 | 0.536
| Red thrombus | 0.3 | 0.301 | 0.373
| White thrombus | 0.231 | 0.211 | 0.233
| Dissection | 0.017 | 0.00002 | 0
| Plaque rupture | 0.33 | 0.256 | 0.32


In this case, we see a noticeable increase in the last 2D model, specially for calcium. 


### Results on test set (frame-level)


| ROI  | 2D model 1st dataset | 2D model 2nd dataset | 2D model 3rd dataset | 3D model sparse
| ------------- | -------------- | -------------- | -------------- | -------------- 
| Lumen  | 0.973 | 0.974 | 0.981
| Guidewire  | 0.928 | 0.93 | 0.941
| Wall | 0.872 | 0.889 | 0.893
| Lipid | 0.415 | 0.465 | 0.553
| Calcium | 0.258 | 0.258 | 0.507
| Media | 0.736 | 0.746 | 0.773
| Catheter | 0.987 | 0.988 | 0.99
| Sidebranch | 0.521 | 0.554 | 0.599
| Red thrombus | 0 | 0.032 | 0.093
| White thrombus | 0 | 0 | 0
| Dissection | 0 | 0 | NaN
| Plaque rupture | 0.321 | 0.368 | 0.377

Note that for the test set, there are no frames with white thrombus or dissections, meaning that the best prediction for these regions would be NaN (i.e no false positives with white thrombus or dissection). Again, we see a great increase in lipid and calcium from the second dataset to the third dataset. We think that the different pre-processing techniques may have played an important role in the outcome of the third model


### Results on test set (pullback-level)

| ROI  | 2D model 1st dataset | 2D model 2nd dataset | 2D model 3rd dataset | 3D model sparse
| ------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.981 | 0.981 | 0.986
| Guidewire  | 0.929 | 0.931 | 0.942
| Wall | 0.885 | 0.9 | 0.903
| Lipid | 0.649 | 0.67 | 0.655
| Calcium | 0.43 | 0.498 | 0.598
| Media | 0.77 | 0.779 | 0.799
| Catheter | 0.987 | 0.988 | 0.990
| Sidebranch | 0.586 | 0.646 | 0.708
| Red thrombus | 0 | 0.122 | 0.094
| White thrombus | 0 | NaN | 0
| Dissection | 0 | NaN | NaN
| Plaque rupture | 0.316 | 0.37 | 0.378 


### Lipid arc DICE

Inspired by the approaches in the study by [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1), we calculated the DICE scores for the lipid arc. This way, we obtain a more insightful measure to asses the model performance (the previous DICE were computed pixel-label). The following table shows these DICE scores for the test set using the prediction given by the three 2D models that we have up to now. An average over the DICE scores for each frame is shown.

Model | Lipid arc fram-level | Lipid arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.666 | 0.797
| Model 2 | 0.759 | 0.832
| Model 3 | 0.767 | 0.827

### Post processing results

For the post-processing measurements, we perfomed a Bland-Altman analysis in order to find the agreement between manual and automatic segmentations.

| Model  | FCT (mean diff / SD) (µm) | Lipid arc (mean diff / SD) (º)
| ------------- | -------------- | -------------- 
| 1  | 37.96 ± [230, -150] | 10.57 ± [80, -60]
| 2  | 35.88 ± [250,  -180] | 6.06 ± [68, -56]
| 3  | 28.51 ± [190,  -130] | 1.71 ± [54, -50]


## TODO:
 - See manual measurements in detail
 - Solve problem with DA 
 - Do training with the selective volumes (3 frames and/or k-neighbors approach)
 - Get ideas from [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1) and [Wang et al. (2012)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3370980/) for maybe more exact post processing measurements.
 - Dive into probability maps for uncertainty estimation

