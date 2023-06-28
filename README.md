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
| First dataset  | 49/14 (1 EST-NEMC, 25 AMPH, 3 HMC, 24 ISALA, 10 RADB)  | 56/15  | 783/195
| Second dataset  | 75/14 (1 EST-NEMC, 28 AMPH, 3 HMC, 24 ISALA, 33 RADB)  | 88/15  | 1215/195
| Third dataset  | 100/14 (1 EST-NEMC, 34 AMPH, 3 HMC, 24 ISALA, 52 RADB)  | 118/15  | 1649/195
| Fourth dataset  | 112/14 (1 EST-NEMC, 34 AMPH, 3 HMC, 24 ISALA, 64 RADB)  | 134/15  | 1846/195


We show the regions of interest (ROIs) that the algorithm segments. With the aim to understand better the dataset, we obtained the distribution for each ROI among the three datasets that were used in the study. These values can be seen in the following table. 

| ROI  | First dataset (frames/pullbacks)(%) | Second dataset (frames/pullbacks) (%) | Third dataset (frames/pullbacks)(%) | Fourth dataset (frames/pullbacks)(%) | Test set (frames/pullbacks)(%)
| ------------- | ------------- | ------------- | ------------- | ------------- | -------------
| Lumen  | - | - | - | - | -
| Guidewire  | - | - | - | - | -
| Wall | - | - | - | - | -
| Lipid | 51.08 / 98.21 | 46.74 / 97.72 | 47.96 / 96.61 | 48.26 / 97.01 | 51.79 / 100
| Calcium | 27.58 / 83.92 | 27.07 / 81.81 | 31.59 / 83.05 | 32.12 / 84.32 | 17.95 / 86.66
| Media | 94.89 / 100 | 96.21 / 100 | 94.9 / 100 | 95.07 / 100 | 98.97 / 100
| Catheter | - | - | - | - | - | -
| Sidebranch | 13.79 / 85.71 | 14.97 / 89.77 | 15.46 / 89.83 | 15.81 / 88.81 | 19.48 / 93.33 
| Red thrombus | 6.89 / 26.78 | 5.67 / 23.86 | 6.67 / 24.57 | 6.07 / 23.13 | 0.51 / 6.66
| White thrombus | 5.61 / 28.57 | 4.53 / 23.86 | 5.45 / 27.96 | 4.87 / 24.62 | 5.64 / 6.66
| Dissection | 0.76 / 5.35 | 0.49 / 3.41 | 0.36 / 2.54 | 0.32 / 2.23 | 4.1 / 6.66
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 7.09 / 20.33 | 6.44 / 19.40 | 2.56 / 13.33


Note that the lumen, guidewire, wall and catheter are present in every frame of the datatset.


## Preprocessing

The general preprocessing consisted of reshaping the images to a common size, which was (704, 704) and applying a circular mask to each slice. This is because each slice contains a watermark by Abbott with a small scale bar, and we do not want our algorithm to learn from this information.

### 2D approach

For the 2D approach, the slices that did not contain any label were omitted. Thus, each slice for every pullback in the dataset was saved to a single NifTI file. In addition, each channel in the slice (RGB values) was saved separately as well, obtaining 3 files for each frame in the pullback. Similiary, each segmentation frame was saved in a different NifTI file. In this case, the segmentation is 1-dimensional, so there was no need to create a file for each dimension.

For the first and second training, a linear interpolation resampler was used for both segmentations and images. In the case of the images, a circular mask with radius 340 was applied. Next, each 2D frame was converted to a pseudo 3D scan by including and extra dimension of shape 1, having a final shape of (1, 704, 704). Finally, the spacing and direction of the frame was set to (1.0, 1.0, 1.0) and (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), respectively.

For the third training, a nearest neighbor interpolation sampler was used, since the liner interpolator introduces more artifacts in the frame. After this step, due to missing segmentations on the edge of the frame, a circular mask with radius 346 was applied to both images and segmentation, so the overlap between them is now perfect. Again, the frames were converted to pseudo 3d scans. However, the spacing of the frame was changed to (1.0, 1.0, 999.0), to avoid possible conflicts with calculating the transpose of the image.

In the fourth training, we fixed the spacing of certain images. The raw images/segmentations are not all the same size, and sometimes the spacing is also altered, leading to getting a zoomed-out version for these cases. For this training we fixed this so all images and segmentations have the same zoom-in. For that, the images were downloaded again from Ultreon (Abbott software) with the correct zoom-in and the segmentations were manually changed using code.

### 3D approach

For the 3D version of the nnUNet, a sparse trainer was initially used. In this case, the loss function is computed using slices that contain annotations in each 3D volume (DC + CE loss). The frames that do not contain any label have a segmentation map that only contains -1, in order for the algorithm to detect unlabeled data. The preprocessing steps are very similar to the 2D model (fourth training), in which each pullback is separated into its RGB values and each volume is saved separately in different NifTI files. Then, the main difference is that now whole 3D volumes are saved, rather than single 2d frames.

(Write about that didnt work and used approaches)

### Pseudo 3D approach

In this case, we still made use of the 2D nnUNet. However, for each frame with annotation, be sampled k frames before and k frames after in order to store some spatial information. We included these frames in the training as modalities for the frame with annotation. That is, we stored the RGB channels for each neighbour frame. If the annotation is the first frame, then the frame(s) before is simply a black image (array with zeros). A nice study by [Chu et al. (2021)](https://eurointervention.pcronline.com/article/automatic-characterisation-of-human-atherosclerotic-plaque-composition-from-intravascular-optical-coherence-tomography-using-artificial-intelligence) used a similar approach



## Training

### nn-UNet

The no-new UNet (nnUNet) is based on the well-known UNet architecture. In this model, an encoder part downsizes the input and increases the number of feature channels, followed by a decoder that takes upsamples the feature maps and reconstructs the original size of the input image. These enconder and decoder networks are also connected by skip connections that allow the decoder to use high-resolution features from the encoder.

The problem with this model is that it needs a very specific input settings and preprocessing the data can be a tedious task. That is why not only the nnUNet uses the U-Net architecture, but also it automatically configures itself, including the preprocessing, network architecture, training and post-processing for any task and data. Hence, while achieving state-of-the-art performances in different tasks, nnUNet adds a systematic framework that can overcome problems and limitations during manual configurations. 

For more information on the nnUNet architecture and processes, see the [original repository](https://github.com/MIC-DKFZ/nnUNet), the [original paper](https://www.nature.com/articles/s41592-020-01008-z) and the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf).

### 2D training

Four 2D nnUNets have been trained, each one with the peculiarities in the input as stated in the preprocessing steps. For these trainings, the parameters and architecture for the nnUNet were the same (batch size = 4, patch size = (768, 768) and adaptive learning rate = 0.01). 

A grayscale 2D model has also been trained using class weights to account for class imbalance, which were obtained by calculating the frequency of each class in the train set pixel level.

### Pseudo 3D training

We have trained three pseudo 3D nnUNets with k=1, k=2 and k=3. For k=2 and k=3, we employed class weights and a weighted loss function (DC loss is weighted more to account for this class imbalance as well, with DC = 3 and CE = 0.8).

Currently, we are doing a pseudo 3D training with k=7, using class weights but withou the weighted loss function (similar as with the 2D grayscale training)

## Post-processing

For the post-processing, we desgined algrithms that perform measurements on the calcium and lipid regions:

### Lipid

An algorithm that automatically measures the fibrous cap thickness (FCT: thickness of the wall that delimitates the lipid and the lumen) and the lipid arc was developed. A plaque is usually deemed vulnerable when a thin-cap fibroatheroma (TCFA) appears or there are either cap rupture or thrombus formation. In the case of TCFA, this occurs when there is a lipd arc ≥ 90º and a FCT < 65 µm. That is why the correct measurement of these two values is very important for the correct treatment of the patient.

![Figure 2. Example of nnUNet prediction (left) with the measured lipid arc and cap thickness (right)](assets/lipid_post_proc_img.png)


### Calcium

Similarly, we also developed an algorithm that performs measurements in the calcium region. The script measures the calcium depth (similar as FCT), calcium thickness (the thickness of the biggest calcium plaque, perpendicular to the lumen) and calcium arc (similar as the lipid arc). In this case, the amount of calcium would indicate that the lesion there should be prepared before treating it with a stent. These values are calcium arc > 180º (score of 2 points), thickness > 0.5 mm (1 point). Another parameter which is not included is the calcium length (length of the calcium in the longitudinal axis), which  has a threshold of > 5 mm (1 point). This gives a calcium score of 0-4 points. See [Fujino et al.](https://pubmed.ncbi.nlm.nih.gov/29400655/) for more information.

![Figure 3. Example of nnUNet prediction (left) with the calcium measurements (right)](assets/calcium_post_proc_img.png)

## Results

We obtained several metrics (accuracy, recall, jaccard, etc), but we only diplay the DICE scores for each one of the regions segmented. For the 2D models, the DICE scores computed per frame are showed. In order to accurately compare the 2D models and the 3D model, the DICE scores for the 2D models were computed pullback-wise (i.e the values of the confusion matrix are computed using every pixel in the pullback, rather than for every frame independently). 

### Results of best cross-validation model

#### 2D models

| ROI  | Model 1 | Model 2 | Model 3 | Model 4 | Model 8 (grayscale)
| ------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.977 | 0.979 | 0.987 | 0.987 | 0.985
| Guidewire  | 0.919 | 0.924 | 0.947 | 0.946 | 0.933
| Wall | 0.883 | 0.892 | 0.901 | 0.899 | 0.884
| Lipid | 0.491 | 0.517 | 0.569 | 0.578 | 0.568
| Calcium | 0.426 | 0.447 | 0.589 | 0.604 | 0.569
| Media | 0.746 | 0.762 | 0.772 | 0.765 | 0.734
| Catheter | 0.981 | 0.984 | 0.992 | 0.992 | 0.988
| Sidebranch | 0.441 | 0.461 | 0.533 | 0.535 | 0.518
| Red thrombus | 0.436 | 0.463 | 0.479 | 0.486 | 0.456
| White thrombus | 0.321 | 0.33 | 0.378 | 0.393 | 0.359
| Dissection | 0.06 | 0.0004 | 0 | 0 | 0.241
| Plaque rupture | 0.471 | 0.429 | 0.542 | 0.527 | 0.521

#### Pseudo 3D models

| ROI  | Model 5 (k=1) | Model 6 (k=2) | Model 7 (k=3) | Model 9 (k=7 grayscale)
| ------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.987 | 0.986 | 0.987 | 0.985
| Guidewire  | 0.947 | 0.941 | 0.941 | 0.934
| Wall | 0.899 | 0.89 | 0.891 | 0.884
| Lipid | 0.579 | 0.583 | 0.581 | 0.576
| Calcium | 0.603 | 0.598 | 0.58 | 0.577
| Media | 0.769 | 0.76 | 0.759 | 0.736
| Catheter | 0.992 | 0.991 | 0.991 | 0.988
| Sidebranch | 0.546 | 0.532 | 0.532 | 0.497
| Red thrombus | 0.459 | 0.455 | 0.454 | 0.464
| White thrombus | 0.382 | 0.418 | 0.404 | 0.389
| Dissection | 0 | 0.247 | 0.293 | 0.247
| Plaque rupture | 0.5541 | 0.512 | 0.523 | 0.542


### Results on test set (frame-level)

#### 2D models


| ROI  | Model 1 | Model 2 | Model 3 | Model 4 | Model 8 (grayscale)
| ------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.976 | 0.979 | 0.981 | 0.980 | 0.977
| Guidewire  | 0.929 | 0.929 | 0.952 | 0.953 | 0.941
| Wall | 0.876 | 0.888 | 0.896 | 0.898 | 0.885 
| Lipid | 0.538 | 0.616 | 0.626 | 0.638 | 0.629 
| Calcium | 0.286 | 0.304 | 0.536 | 0.502 | 0.489 
| Media | 0.742 | 0.760 | 0.781 | 0.784 | 0.737 
| Catheter | 0.989 | 0.99 | 0.99 | 0.99 | 0.987
| Sidebranch | 0.483 | 0.511 | 0.534 | 0.494 | 0.471
| Red thrombus | 0 | 0.014 | 0.023 | 0.092 | 0.022
| White thrombus | 0.198 | 0.227 | 0.288 | 0.299 | 0.242
| Dissection | 0.0004 | 0 | 0 | 0 | 0
| Plaque rupture | 0.343 | 0.326 | 0.252 | 0.379 | 0.24

#### Pseudo 3D models

| ROI  | Model 5 (k=1) | Model 6 (k=2) | Model 7 (k=3) | Model 9 (k=7 grayscale)
| ------------- | -------------- | -------------- | -------------- | -------------- 
| Lumen  | 0.974 | 0.978 | 0.978 | 0.977
| Guidewire  | 0.952 | 0.949 | 0.949 | 0.94
| Wall | 0.879 | 0.89 | 0.888 | 0.885
| Lipid | 0.621 | 0.644 | 0.645 | 0.649
| Calcium | 0.506 | 0.492 | 0.548 | 0.525
| Media | 0.744 | 0.767 | 0.767 | 0.739
| Catheter | 0.989 | 0.988 | 0.989 | 0.986
| Sidebranch | 0.502 | 0.507 | 0.531 | 0.472
| Red thrombus | 0.025 | 0.047 | 0.05 | 0.015
| White thrombus | 0.037 | 0.245 | 0.226 | 0.224
| Dissection | 0 | 0 | 0 | 0
| Plaque rupture | 0.314 | 0.261 | 0.278 | 0.359    


<!-- ### Results on test set (pullback-level)

| ROI  | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 (k=1)
| ------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.976 | 0.982 | 0.985 | 0.984 | 0.981
| Guidewire  | 0.928 | 0.928 | 0.951 | 0.951 | 0.952
| Wall | 0.887 | 0.897 | 0.902 | 0.904 | 0.896
| Lipid | 0.696 | 0.713 | 0.701 | 0.713 | 0.705
| Calcium | 0.512 | 0.528 | 0.596 | 0.562 | 0.638
| Media | 0.777 | 0.781 | 0.797 | 0.798 | 0.779
| Catheter | 0.989 | 0.989 | 0.989 | 0.989 | 0.989
| Sidebranch | 0.668 | 0.728 | 0.764 | 0.749 | 0.707
| Red thrombus | 0 | 0.044 | 0.047 | 0.234 | 0.111
| White thrombus | 0.137 | 0.278 | 0.21 | 0.232 | 0.035
| Dissection | 0.0004 | 0 | 0 | 0 | 0
| Plaque rupture | 0.202 | 0.246 | 0.252 | 0.252 | 0.259 -->


### Lipid arc DICE

Inspired by the approaches in the study by [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1), we calculated the DICE scores for the lipid arc. This way, we obtain a more insightful measure to asses the model performance (the previous DICE were computed pixel-label). The following table shows these DICE scores for the test set using the prediction given by the three 2D models that we have up to now. An average over the DICE scores for each frame is shown.

Model | Lipid arc frame-level | Lipid arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.687 | 0.838
| Model 2 | 0.765 | 0.872
| Model 3 | 0.786 | 0.883
| Model 4 | 0.777 | 0.887
| Model 5 | 0.754 | 0.861
| Model 6 | 0.761 | 0.867
| Model 7 | 0.762 | 0.864
| Model 8 | 0.751 | 0.874
| Model 9 | 0.775 | 0.861 


### Calcium arc DICE

Similar as with the lipid arc, we computed the DICE for the detected arc of the biggest calcium region that appears in the frame. We also computed the DICE per frame and per pullback.

Model | Calcium arc frame-level | Calcium arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.506 | 0.612
| Model 2 | 0.564 | 0.601
| Model 3 | 0.651 | 0.694
| Model 4 | 0.588 | 0.695
| Model 5 | 0.632 | 0.758
| Model 6 | 0.585 | 0.699
| Model 7 | 0.611 | 0.673
| Model 8 | 0.575 | 0.681
| Model 9 | 0.613 | 0.639

### Post processing results

For the post-processing results, we report the Bland-Altman analysis and intra-class correlation (ICC) for the measurements on the predictions and the manual measurements.

#### Lipid measurements

| Model  | FP | FN | FCT (mean diff / SD) (µm) | Lipid arc (mean diff / SD) (º) | FCT ICC(2,1) | Lipid arc ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- 
| 1  | 21 | 6 | 22.32 ± 78.1 | 14.99 ± 36.64 | 0.77 | 0.791
| 2  | 11 | 5 | 15.51 ± 68.02 | 8.69 ± 29.74 | 0.828 | 0.857
| 3  | 11 | 3 | 28.56 ± 94.68 | 3.40 ± 26.6 | 0.685 | 0.893
| 4  | 13 | 2 | 20.58 ± 91.33 | 3.59 ± 27.91 | 0.717 | 0.883
| 5  | 14 | 3 | 28.76 ± 111.55 | 4.39 ± 32.36 | 0.58 | 0.852
| 6  | 12 | 3 | 25.92 ± 121.74 | -0.73 ± 30.39 | 0.571 | 0.866
| 7  | 12 | 4 | 26.8 ± 99.37 | -1.59 ± 27.66 | 0.666 | 0.886
| 8  | 13 | 4 | 32.6 ± 133.7 | 4.1 ± 30.27 | 0.616 | 0.865
| 9  | 10 | 3 | 45.02 ± 142.27 | 3.87 ± 33.27 | 0.541 | 0.847


#### Calcium measurements


| Model | FP | FN | Depth (mean diff / SD) (µm) | Arc (mean diff / SD) (º) | Thickness (mean diff /SD) (µm) | Depth ICC(2,1) | Calcium arc ICC(2,1) | Thickness ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- 
| 1  | 12 | 6 | -3.89 ± 67.62 | -13.28 ± 21.35 | -107.82 ± 211.24 | 0.754 | 0.728 | 0.747 
| 2  | 9 | 5 | -0.2 ± 42.63 | -12.7 ± 22.22 | -55.56 ± 163.63 | 0.917 | 0.727 | 0.859
| 3  | 7 | 2 | 11.63 ± 60.39 | -12.15 ± 20.37 | -79.18 ± 167.13 | 0.875 | 0.727 | 0.848
| 4  | 10 | 3 | 9.59 ± 50.21 | -11.59 ± 22.31 | -84.31 ± 162.16 | 0.913 | 0.719 | 0.851
| 5  | 7 | 2 | 15 ± 56.21 | -11.06 ± 22.58 | -61.85 ± 174.21 | 0.889 | 0.737 | 0.843
| 6  | 9 | 5 | 4.5 ± 53.93 | -9.9 ± 22.74 | -45.53 ± 154.81 | 0.9 | 0.739 | 0.879
| 7  | 8 | 4 | 0.03 ± 45.19 | -9.39 ± 22.54 | -49.52 ± 190.82 | 0.929 | 0.748 | 0.82
| 8  | 8 | 5 | -9.93 ± 40.94 | -9.16 ± 26.56 | -45.53 ± 193.71 | 0.922 | 0.664 | 0.808
| 9  | 6 | 5 | -2 ± 42.81 | -6.96 ± 25.26 | -38.2 ± 157.69 | 0.918 | 0.72 | 0.874


## TODO:
 - Train pseudo 3d (+- 7 frames and grayscale)
 - Probability maps and uncertainty estimation: see losses functions and correlation with DICE
 - Model architecure and weights to see feature map (explainability)
 - More post processing: connected component analysis and morph operations

