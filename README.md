# CardiacOCT project

## Project description

Acute myocardial infarction (MI) remains as one of the leading causes of mortality in the developed world. Despite huge advances in treating this condition such as the use of angiograms to locate the occluded artery or coronary angioplasty, there is still a debate on whether to treat certain lesions found during coronary angiography or not, since it is difficult to predict which plaques would have a worse outcome for the patient’s health. Imaging modalities such as intracoronary optical coherence tomography (OCT) provide a better comprehension of plaque characteristics and it can help surgeons to better asses these lesions, improving patient’s outcome.

In this project, an automatic segmentation model will be designed for intracoronary OCT scans in order to asses for plaque vulnerability and detect other abnormalities such as white or red thrombus or plaque rupture. Specifically, a nnUNet (no-new-UNet) that works with sparse annotated data will be designed. Initially, the model will be trained on singles frames that contain a corresponding segmentation map, that is, the model works in a supervised manner. Next, in order to account for the sparse annotations, a 3D UNet will be trained in a semi-supervised manner. After the models have been trained, several automatic post-processing techninques for lipid arc and cap thickness measurement will be implemented. Moreover, an uncertainty estimation model will be designed in order to detect unreliable segmentations and add more value to the algorithm's output.


<p float="left" align="center">
<img src="https://user-images.githubusercontent.com/37450737/220990629-a658d95a-8c3b-4fb9-9289-d44a6c1d26d3.png" width=35% height=35%>
<img src="https://user-images.githubusercontent.com/37450737/220990554-bb602dce-e69f-4a0e-ad8e-163e607415e6.png" width=35% height=35%>
<figcaption> Figure 1. Example of intracoronary OCT frame (left) with its corresponding manual segmentation (right) </figcaption>
<p>

           
## Dataset

The intracoronary OCT dataset used in this study is a collection of OCT scans from 5 different medical centers: Isala Zwole (ISALA), Amphia Hospital (AMPH), ?? (NEMC), Hague Medical Centrum (HMC) and RadboudUMC (RADB).

Since the manually labelling of OCT frames is a very time consuming task for annotators, not all scans were included for the training. In particular, the standard methodology is to label each 40th frame in the scan, unless there are some regions in other frames that are necessarily to label. Thus, frames that contain some degree of labelling were included for the training. 

| Dataset  | Nº of patients (train/test) | Nº of pullbacks (train/test) | Nº of annotated frames (train/test)
| ------------- | ------------- | -------------  | -------------
| First dataset  | 49/13 (1 EST-NEMC, 24 AMPH, 3 HMC, 24 ISALA, 10 RADB)  | 56/14  | 783/163
| Second dataset  | 75/13 (1 EST-NEMC, 27 AMPH, 3 HMC, 24 ISALA, 33 RADB)  | 88/14  | 1215/162
| Third dataset  | 100/13 (1 EST-NEMC, 33 AMPH, 3 HMC, 24 ISALA, 52 RADB)  | 118/14  | 1649/162 


We show the regions of interest (ROIs) that the algorithm segments. With the aim to understand better the dataset, we obtained the distribution for each ROI among the three datasets that were used in the study. These values can be seen in the following table. 

| ROI  | First dataset (frames/pullbacks)(%) | Second dataset (frames/pullbacks) (%) | Third dataset (frames/pullbacks)(%) | Test set (frames/pullbacks)(%)
| ------------- | ------------- | ------------- | ------------- | -------------
| Lumen  | - | - | - | -
| Guidewire  | - | - | - | -
| Wall | - | - | - | -
| Lipid | 51.08 / 98.21 | 46.74 / 97.72 | 47.96 / 96.61 | 48.14 / 92.85 
| Calcium | 27.58 / 83.92 | 27.07 / 81.81 | 31.59 / 83.05 | 16.66 / 71.42
| Media | 94.89 / 100 | 96.21 / 100 | 94.9 / 100 | 99.38 / 100
| Catheter | - | - | - | - | -
| Sidebranch | 13.79 / 85.71 | 14.97 / 89.77 | 15.46 / 89.83 | 16.67 / 71.42 
| Red thrombus | 6.89 / 26.78 | 5.67 / 23.86 | 6.67 / 24.57 | 0.61 / 7.14
| White thrombus | 5.61 / 28.57 | 4.53 / 23.86 | 5.45 / 27.96 | 0 / 0
| Dissection | 0.76 / 5.35 | 0.49 / 3.41 | 0.36 / 2.54 | 0 / 0
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 7.09 / 20.33 | 3.08 / 14.28


Note that the lumen, guidewire, wall and catheter are present in every frame of the datatset.


## Preprocessing

The general preprocessing consisted of reshaping the images to a common size, which was (704, 704) and applying a circular mask to each slice. This is because each slice contains a watermark by Abbott with a small scale bar, and we do not want our algorithm to learn from this information.

### 2D approach

For the 2D approach, the slices that did not contain any label were omitted. Thus, each slice for every pullback in the dataset was saved to a single NifTI file. In addition, each channel in the slice (RGB values) were saved separately as well, obtaining 3 files for each frame in the pullback. Similiary, each segmentation frame was saved in a different NifTI file. In this case, the segmentation is 1-dimensional, so there was no need to create a file for each dimension.

For the first and second training, a linear interpolation resampler was used for both segmentations and images. In the case of the images, a circular mask with radius 340 was applied. Next, each 2D frame was converted to a pseudo 3D scan by including and extra dimension of shape 1, having a final shape of (1, 704, 704). Finally, the spacing and direction of the frame was set to (1.0, 1.0, 1.0) and (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), respectively.

For the third training, a nearest neighbor interpolation sampler was used, since the liner interpolator introduces more artifacts in the frame. After this step, due to missing segmentations on the edge of the frame, a circular mask with radius 346 was applied to both images and segmentation, so the overlap between them is now perfect. Again, the frames were converted to pseudo 3d scans. However, the spacing of the frame was changed to (1.0, 1.0, 999.0), to avoid possible conflicts with calculating the transpose of the image.

### 3D approach

For the 3D version of the nnUNet, a sparse trainer was used. In this case, the loss function is computed using slices that contain annotations in each 3D volume. The frames that do not contain any label have a segmentation map that only contains -1, in order to the algorithm to detect unlabeled data. The preprocessing steps are very similar to the 2D model (third training), in which each pullback is separated into its RGB values and each volume is saved separately in different NifTI files. Then, the main difference is that now whole 3D volumes are saved, rather than single 2d frames.

(Expand more when training)



## Results

We obtained several metrics (accuracy, recall, jaccard, etc), but we only diplay the DICE scores for each one of the regions segmented.

### Results of best cross-validation model


| ROI  | 2D model 1st dataset | 2D model 2nd dataset | 2D model 3rd dataset | 3D sparse model
| ------------- | -------------- | -------------- | -------------- | -------------- 
| Lumen  | 0.981
| Guidewire  | 0.927
| Wall | 0.892
| Lipid | 0.341
| Calcium | 0.162
| Media | 0.716
| Catheter | 0.985
| Sidebranch | 0.105 
| Red thrombus | 0.043 
| White thrombus | 0.014  
| Dissection | 0.0016 
| Plaque rupture | 0.039 


### Results on test set


| ROI  | 2D model 1st dataset | 2D model 2nd dataset | 3D sparse model 2nd dataset 
| ------------- | -------------- | -------------- | --------------
| Lumen  | 0.973 | 0.974
| Guidewire  | 0.928 | 0.93
| Wall | 0.872 | 0.889
| Lipid | 0.415 | 0.465
| Calcium | 0.258 | 0.258
| Media | 0.736 | 0.746
| Catheter | 0.987 | 0.988
| Sidebranch | 0.521  | 0.554
| Red thrombus | 0 | 0.032
| White thrombus | 0 | 0 
| Dissection | 0 | 0
| Plaque rupture | 0.321 | 0.368

Note that for the test set, there are no frames with white thrombus or dissections, meaning that the best prediction for these regions would be NaN (i.e no false positives with white thrombus or dissection)


## TODO:
 - Train data with third dataset
 - Continue building metrics and Excel files with distributions (also make them fancier)
 - Solve NaN problem with 3d version on third dataset --> impute -1 once preprocessing is done, check again with new resampling
 - Figure out post-processing techniques for lipid arc and cap thickness measurements (using dynammic programming + semantic segmentation of lipid, lumen, intima, etc). 
 For this, check [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1) and [Wang et al. (2012)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3370980/)

