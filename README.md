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
| First dataset  | 49/14 (1 EST-NEMC, 25 AMPH, 3 HMC, 24 ISALA, 10 RADB)  | 56/15  | 783/183
| Second dataset  | 75/14 (1 EST-NEMC, 28 AMPH, 3 HMC, 24 ISALA, 33 RADB)  | 88/15  | 1215/183
| Third dataset  | 100/14 (1 EST-NEMC, 34 AMPH, 3 HMC, 24 ISALA, 52 RADB)  | 118/15  | 1649/183 
| Fourth dataset  | 112/14 (1 EST-NEMC, 34 AMPH, 3 HMC, 24 ISALA, 64 RADB)  | 134/15  | 1846/183 


We show the regions of interest (ROIs) that the algorithm segments. With the aim to understand better the dataset, we obtained the distribution for each ROI among the three datasets that were used in the study. These values can be seen in the following table. 

| ROI  | First dataset (frames/pullbacks)(%) | Second dataset (frames/pullbacks) (%) | Third dataset (frames/pullbacks)(%) | Fourth dataset (frames/pullbacks)(%) | Test set (frames/pullbacks)(%)
| ------------- | ------------- | ------------- | ------------- | ------------- | -------------
| Lumen  | - | - | - | - | -
| Guidewire  | - | - | - | - | -
| Wall | - | - | - | - | -
| Lipid | 51.08 / 98.21 | 46.74 / 97.72 | 47.96 / 96.61 | 48.26 / 97.01 | 51.91 / 93.33 
| Calcium | 27.58 / 83.92 | 27.07 / 81.81 | 31.59 / 83.05 | 32.12 / 84.32 | 16.93 / 73.33
| Media | 94.89 / 100 | 96.21 / 100 | 94.9 / 100 | 95.07 / 100 | 99.45 / 100
| Catheter | - | - | - | - | - | -
| Sidebranch | 13.79 / 85.71 | 14.97 / 89.77 | 15.46 / 89.83 | 15.81 / 88.81 | 18.03 / 73.33 
| Red thrombus | 6.89 / 26.78 | 5.67 / 23.86 | 6.67 / 24.57 | 6.07 / 23.13 | 0.55 / 6.66
| White thrombus | 5.61 / 28.57 | 4.53 / 23.86 | 5.45 / 27.96 | 4.87 / 24.62 | 6.01 / 6.66
| Dissection | 0.76 / 5.35 | 0.49 / 3.41 | 0.36 / 2.54 | 0.32 / 2.23 | 4.37 / 6.66
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 7.09 / 20.33 | 6.44 / 19.40 | 2.73 / 13.33


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

In this case, we still made use of the 2D nnUNet. However, for each frame with annotation, be sampled k frames before and k frames after in order to store some spatial information. We included these frames in the training as modalities for the frame with annotation. That is, we stored the RGB channels for each neighbour frame. If the annotation is the first frame, then the frame(s) before is simply a black image (array with 0s). A nice study by [Chu et al. (2021)](https://eurointervention.pcronline.com/article/automatic-characterisation-of-human-atherosclerotic-plaque-composition-from-intravascular-optical-coherence-tomography-using-artificial-intelligence) used a similar approach

In particular, we have tried with one frame before and one after (getting a total of 9 modalities). Currently, we are sampling 7 frames before and 7 after, getting a total of 45 files for a single annotation. Moreover, for this new training, we are employing class weights for account the class imbalanace and a weighted loss function (DC loss is weighted more to account for this clas imbalance as well). This training is still under construction.

## Training

### nn-UNet

The no-new UNet (nnUNet) is based on the well-known UNet architecture. In this model, an encoder part downsizes the input and increases the number of feature channels, followed by a decoder that takes upsamples the feature maps and reconstructs the original size of the input image. These enconder and decoder networks are also connected by skip connections that allow the decoder to use high-resolution features from the encoder.

The problem with this model is that it needs a very specific input settings and preprocessing the data can be a tedious task. That is why not only the nnUNet uses the U-Net architecture, but also it automatically configures itself, including the preprocessing, network architecture, training and post-processing for any task and data. Hence, while achieving state-of-the-art performances in different tasks, nnUNet adds a systematic framework that can overcome problems and limitations during manual configurations. 

For more information on the nnUNet architecture and processes, see the [original repository](https://github.com/MIC-DKFZ/nnUNet), the [original paper](https://www.nature.com/articles/s41592-020-01008-z) and the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf).

## Post-processing

For the post-processing, we desgined algrithms that perform measurements on the calcium and lipid regions:

### Lipid

An algorithm that automatically measures the fibrous cap thickness (FCT: thickness of the wall that delimitates the lipid and the lumen) and the lipid arc was developed. A plaque is usually deemed vulnerable when a thin-cap fibroatheroma (TCFA) appears or there are either cap rupture or thrombus formation. In the case of TCFA, this occurs when there is a lipd arc ≥ 90º and a FCT < 65 µm. That is why the correct measurement of these two values is very important for the correct treatment of the patient.

![Figure 2. Example of nnUNet prediction (left) with the measured lipid arc and cap thickness (right)](assets/lipid_post_proc_img.png)


### Calcium

Similarly, we also developed an algorithm that performs measurements in the calcium region. The script measures the calcium depth (similar as FCT), calcium thickness (the thickness of the biggest calcium plaque, perpendicular to the lumen) and calcium arc (similar as the lipid arc). In this case, the amount of calcium would indicate that the lesion there should be prepared before treating it with a stent. These values are calcium arc > 180º (score of 2 points), thickness > 0.5 mm (1 point). Another parameter which is not included is tbe calcium length (length of the calcium in the longitudinal axis), which  has a threshold of > 5 mm (1 point). This gives a calcium score of 0-4 points. See [Fujino et al.](https://pubmed.ncbi.nlm.nih.gov/29400655/) for more information.

![Figure 3. Example of nnUNet prediction (left) with the calcium measurements (right)](assets/calcium_post_proc_img.png)

## Results

We obtained several metrics (accuracy, recall, jaccard, etc), but we only diplay the DICE scores for each one of the regions segmented. For the 2D models, the DICE scores computed per frame are showed. In order to accurately compare the 2D models and the 3D model, the DICE scores for the 2D models were computed pullback-wise (i.e the values of the confusion matrix are computed using every pixel in the pullback, rather than for every frame independently). 

### Results of best cross-validation model

| ROI  | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 (k=1)
| ------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.977 | 0.979 | 0.987 | 0.987 | 0.987
| Guidewire  | 0.919 | 0.924 | 0.947 | 0.946 | 0.947
| Wall | 0.883 | 0.892 | 0.901 | 0.899 | 0.899
| Lipid | 0.491 | 0.517 | 0.569 | 0.578 | 0.579
| Calcium | 0.426 | 0.447 | 0.589 | 0.604 | 0.603
| Media | 0.746 | 0.762 | 0.772 | 0.765 | 0.769
| Catheter | 0.981 | 0.984 | 0.992 | 0.992 | 0.992
| Sidebranch | 0.441 | 0.461 | 0.533 | 0.535 | 0.546
| Red thrombus | 0.436 | 0.463 | 0.479 | 0.486 | 0.459
| White thrombus | 0.321 | 0.33 | 0.378 | 0.393 | 0.382
| Dissection | 0.06 | 0.0004 | 0 | 0 | 0
| Plaque rupture | 0.471 | 0.429 | 0.542 | 0.527 | 0.554


### Results on test set (frame-level)


| ROI  | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 (k=1)
| ------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.976 | 0.978 | 0.981 | 0.981 | 0.974
| Guidewire  | 0.928 | 0.929 | 0.951 | 0.952 | 0.952
| Wall | 0.874 | 0.887 | 0.894 | 0.896 | 0.881
| Lipid | 0.527 | 0.607 | 0.617 | 0.632 | 0.619
| Calcium | 0.282 | 0.291 | 0.529 | 0.491 | 0.514
| Media | 0.749 | 0.761 | 0.783 | 0.785 | 0.756
| Catheter | 0.989 | 0.989 | 0.989 | 0.989 | 0.989
| Sidebranch | 0.443 | 0.491 | 0.514 | 0.47 | 0.487
| Red thrombus | 0 | 0.014 | 0.026 | 0.092 | 0.028
| White thrombus | 0.198 | 0.227 | 0.288 | 0.299 | 0.037
| Dissection | 0.0004 | 0 | 0 | 0 | 0
| Plaque rupture | 0.343 | 0.326 | 0.252 | 0.379 | 0.314


(Write new stuff when done predictions)


### Results on test set (pullback-level)

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
| Plaque rupture | 0.202 | 0.246 | 0.252 | 0.252 | 0.259


### Lipid arc DICE

Inspired by the approaches in the study by [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1), we calculated the DICE scores for the lipid arc. This way, we obtain a more insightful measure to asses the model performance (the previous DICE were computed pixel-label). The following table shows these DICE scores for the test set using the prediction given by the three 2D models that we have up to now. An average over the DICE scores for each frame is shown.

Model | Lipid arc frame-level | Lipid arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.682 | 0.806
| Model 2 | 0.764 | 0.836
| Model 3 | 0.783 | 0.831
| Model 4 | 0.776 | 0.845
| Model 5 | 0.753 | 0.835


### Calcium arc DICE

Similar as with the lipid arc, we computed the DICE for the detected arc of the biggest calcium region that appears in the frame. We also computed the DICE per frame and per pullback.

Model | Calcium arc frame-level | Calcium arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.515 | 0.616
| Model 2 | 0.568 | 0.606
| Model 3 | 0.657 | 0.699
| Model 4 | 0.589 | 0.699
| Model 5 | 0.641 | 0.779

### Post processing results

For the post-processing results, we report the Bland-Altman analysis and intra-class correlation (ICC) for the measurements on the predictions and the manual measurements.

#### Lipid measurements

| Model  | FCT (mean diff / SD) (µm) | Lipid arc (mean diff / SD) (º) | FCT ICC(2,1) | Lipid arc ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- 
| 1  | 27.09 ± 73.69 | 13.03 ± 35.59 | 0.784 | 0.805
| 2  | 19.54 ± 62.47 | 7.16 ± 27.98 | 0.847 | 0.875
| 3  | 34.28 ± 90.48 | 2.77 ± 26.06 | 0.701 | 0.898
| 4  | 26.64 ± 87.29 | 2.23 ± 25.96 | 0.735 | 0.899
| 5  | 38.42 ± 105.46 | 2.84 ± 30.17 | 0.609 | 0.867


#### Calcium measurements


| Model  | Depth (mean diff / SD) (µm) | Arc (mean diff / SD) (º) | Thickness (mean diff /SD) (µm) | Depth ICC(2,1) | Calcium arc ICC(2,1) | Thickness ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- 
| 1  | -10.5 ± 67.99 | -9.31 ± 15.57 | -96.88 ± 215.88 | 0.751 | 0.854 | 0.755 
| 2  | -6.07 ± 40.02 | -9.26 ± 16.88 | -48.63 ± 168.07 | 0.926 | 0.838 | 0.862
| 3  | 10.03 ± 62.12 | -9.53 ± 14.94 | -75.83 ± 171.84 | 0.869 | 0.863 | 0.85
| 4  | 8.28 ± 50.93 | -9.59 ± 17.71 | -79.03 ± 162.41 | 0.91 | 0.812 | 0.862
| 5  | 14 ± 57.53 | -11.27 ± 17.48 | - 72.4 ± 156.54 | 0.887 | 0.809 | 0.871


## TODO:
 - Train pseudo 3d (+- 2 frames)
 - See training with bigger k and using grayscale (problems with OOM)
 - Dive into probability maps for uncertainty estimation and calibration techniques

