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
| **Final dataset**  | 115/15 (1 EST-NEMC, 34 AMPH, 3 HMC, 24 ISALA, 64 RADB)  | 133/15  | 1810/218


We show the regions of interest (ROIs) that the algorithm segments. With the aim to understand better the dataset, we obtained the distribution for each ROI among the three datasets that were used in the study. These values can be seen in the following table. 

<!-- | ROI  | First dataset (frames/pullbacks)(%) | Second dataset (frames/pullbacks) (%) | Third dataset (frames/pullbacks)(%) | Fourth dataset (frames/pullbacks)(%) | Test set (frames/pullbacks)(%)
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
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 7.09 / 20.33 | 6.44 / 19.40 | 2.56 / 13.33 -->

| ROI  | Frames (train %) | Pullbacks (train %) | Frames (test %) | Pullbacks (test %)
| ------------- | ------------- | ------------- | ------------- | ------------- 
| Lumen  | - | - | - | - 
| Guidewire  | - | - | - | - 
| Wall | - | - | - | - 
| Lipid | 48.12 | 97.74 | 51.38 | 93.33
| Calcium | 31.44 | 84.21 | 26.6 | 86.67
| Media | 95.69 | 100 | 93.12 | 100
| Catheter | - | - | - | - | - 
| Sidebranch | 15.91 | 89.47 |19.27 | 93.33
| Red thrombus | 5.19 | 21.8 | 7.34 | 13.33
| White thrombus | 4.97 | 22.56 | 4.13 | 20
| Dissection | 0.77 | 3.01 | 0 | 0 
| Plaque rupture | 6.24 | 19.55 | 2.75 | 6.67


Note that the lumen, guidewire, wall and catheter are present in every frame of the datatset. Moreover, dissections were finally all included in the train set due to the very limited ammount of dissections in the dataset.


## Preprocessing

The general preprocessing consisted of reshaping the images to a common size, which was (704, 704) and applying a circular mask to each slice. This is because each slice contains a watermark by Abbott with a small scale bar, and we do not want our algorithm to learn from this information.

### 2D approach

For the 2D approach, the slices that did not contain any label were omitted. Thus, each slice for every pullback in the dataset was saved to a single NifTI file. In addition, each channel in the slice (RGB values) was saved separately as well, obtaining 3 files for each frame in the pullback. Similiary, each segmentation frame was saved in a different NifTI file. In this case, the segmentation is 1-dimensional, so there was no need to create a file for each dimension.

A nearest neighbor interpolation was used to resample the segmentations. The raw frames were all exported with the (704, 704), so no resampling was needed in this case. Moreover, a circular mask with radius 346 was employed to remove the above mentioned Abbott watermark. Each frame channel was saved independently with shape (1, 704, 704) and spacing (1.0, 1.0, 999.0) and direction (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).

<!-- ### 3D approach

For the 3D version of the nnUNet, a sparse trainer was initially used. In this case, the loss function is computed using slices that contain annotations in each 3D volume (DC + CE loss). The frames that do not contain any label have a segmentation map that only contains -1, in order for the algorithm to detect unlabeled data. The preprocessing steps are very similar to the 2D model (fourth training), in which each pullback is separated into its RGB values and each volume is saved separately in different NifTI files. Then, the main difference is that now whole 3D volumes are saved, rather than single 2d frames. -->

(Write about that didnt work and used approaches)

### Pseudo 3D approach

In this case, we still made use of the 2D nnUNet. However, for each frame with annotation, be sampled k frames before and k frames after in order to store some spatial information. We included these frames in the training as modalities for the frame with annotation. That is, we stored the RGB channels for each neighbour frame. If the annotation is the first frame, then the frame(s) before is simply a black image (array with zeros). A nice study by [Chu et al. (2021)](https://eurointervention.pcronline.com/article/automatic-characterisation-of-human-atherosclerotic-plaque-composition-from-intravascular-optical-coherence-tomography-using-artificial-intelligence) used a similar approach

## Training

### nn-UNet

The no-new UNet (nnUNet) is based on the well-known UNet architecture. In this model, an encoder part downsizes the input and increases the number of feature channels, followed by a decoder that takes upsamples the feature maps and reconstructs the original size of the input image. These enconder and decoder networks are also connected by skip connections that allow the decoder to use high-resolution features from the encoder.

The problem with this model is that it needs a very specific input settings and preprocessing the data can be a tedious task. That is why not only the nnUNet uses the U-Net architecture, but also it automatically configures itself, including the preprocessing, network architecture, training and post-processing for any task and data. Hence, while achieving state-of-the-art performances in different tasks, nnUNet adds a systematic framework that can overcome problems and limitations during manual configurations. 

For more information on the nnUNet architecture and processes, see the [original repository](https://github.com/MIC-DKFZ/nnUNet), the [original paper](https://www.nature.com/articles/s41592-020-01008-z) and the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf).

### Training

4 models were trained: one 2D model and three pseudo 3D models (k=1, k=2 and k=3), all of them RGB. The model hyperparameters were finally kept the same for every trained model: batch size = 4, path size = (768, 768), adaptive learning rate = 0.01, weighted DICE and Cross-Entropy loss.

<!-- Four 2D nnUNets have been trained, each one with the peculiarities in the input as stated in the preprocessing steps. For these trainings, the parameters and architecture for the nnUNet were the same (batch size = 4, patch size = (768, 768) and adaptive learning rate = 0.01).  -->

<!-- A grayscale 2D model has also been trained using class weights to account for class imbalance, which were obtained by calculating the frequency of each class in the train set pixel level. -->
<!-- 
### Pseudo 3D training

We have trained three pseudo 3D nnUNets with k=1, k=2 and k=3. For k=2 and k=3, we employed class weights and a weighted loss function (DC loss is weighted more to account for this class imbalance as well, with DC = 3 and CE = 0.8).

Currently, we are doing a pseudo 3D training with k=7, using class weights but withou the weighted loss function (similar as with the 2D grayscale training) -->

## Post-processing

For the post-processing, we desgined algrithms that perform measurements on the calcium and lipid regions:

### Lipid

An algorithm that automatically measures the fibrous cap thickness (FCT: thickness of the wall that delimitates the lipid and the lumen) and the lipid arc was developed. A plaque is usually deemed vulnerable when a thin-cap fibroatheroma (TCFA) appears or there are either cap rupture or thrombus formation. In the case of TCFA, this occurs when there is a lipd arc ≥ 90º and a FCT < 65 µm. That is why the correct measurement of these two values is very important for the correct treatment of the patient.

![Figure 2. Example of nnUNet prediction (left) with the measured lipid arc and cap thickness (right)](assets/lipid_post_proc_img.png)


### Calcium

Similarly, we also developed an algorithm that performs measurements in the calcium region. The script measures the calcium depth (similar as FCT), calcium thickness (the thickness of the biggest calcium plaque, perpendicular to the lumen) and calcium arc (similar as the lipid arc). In this case, the amount of calcium would indicate that the lesion there should be prepared before treating it with a stent. These values are calcium arc > 180º (score of 2 points), thickness > 0.5 mm (1 point). Another parameter which is not included is the calcium length (length of the calcium in the longitudinal axis), which  has a threshold of > 5 mm (1 point). This gives a calcium score of 0-4 points. See [Fujino et al.](https://pubmed.ncbi.nlm.nih.gov/29400655/) for more information.

![Figure 3. Example of nnUNet prediction (left) with the calcium measurements (right)](assets/calcium_post_proc_img.png)

## Results

We report the DICE (both per frame and pullback), positive predictive value (PPV), negative predictive value (NPV), sensitivity, specificity and Cohen´s Kappa. 

### Results on test set (frame-level)

#### 2D model

| ROI  | DICE per frame | DICE per pullback | PPV | NPV | Sensitivity | Specificity | Kappa
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | --------------
| Lumen  | 0.985 | 0.986 | 1 | NaN | 1 | NaN | NaN
| Guidewire  | 0.939 | 0.943 | 1 | NaN | 1 | NaN | NaN
| Wall | 0.887 | 0.893 | 1 | NaN | 1 | NaN | NaN
| Lipid | 0.599 | 0.695 | 0.809 | 1 | 1 | 0.793 | 0.782
| Calcium | 0.495 | 0.537 | 0.8 | 0.939 | 0.815 | 0.933 | 0.743
| Media | 0.758 | 0.808 | 0.99 | 0.667 | 0.976 | 0.833 | 0.724
| Catheter | 0.99 | 0.99 | 1 | NaN | 1 | NaN | NaN
| Sidebranch | 0.475 | 0.555 | 0.833 | 0.949 | 0.795 | 0.96 | 0.768
| Red thrombus | 0.587 | 0.328 | 0.889 | 0.995 | 0.941 | 0.99 | 0.907
| White thrombus | 0.207 | 0.192 | 1 | 0.967 | 0.364 | 1 | 0.52 
| Dissection | 0 | 0 | 0 | 1 | NaN | 0.968 | 0
| Plaque rupture | 0.155 | 0.08 | 0.375 | 1 | 1 | 0.953 | 0.526

#### Pseudo 3D models


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

| Model | Lipid arc frame-level | Lipid arc pullback-level
| ------------- | -------------- | --------------
| RGB 2D | 0.711 | 0.796



### Calcium arc DICE

Similar as with the lipid arc, we computed the DICE for the detected arc of the biggest calcium region that appears in the frame. We also computed the DICE per frame and per pullback.

Model | Calcium arc frame-level | Calcium arc pullback-level
| ------------- | -------------- | --------------
| Model 1 | 0.61 | 0.678


### Post processing results

For the post-processing results, we report the Bland-Altman analysis and intra-class correlation (ICC) for the measurements on the predictions and the manual measurements.

#### Lipid measurements

| Model  | FP | FN | FCT (mean diff / SD) (µm) | Lipid arc (mean diff / SD) (º) | FCT ICC(2,1) | Lipid arc ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- 
| RGB 2D | 24 | 0 | 4.83 ± 74.42 | 25.36 ± 55.36 | 0.749 | 0.785


#### Calcium measurements


| Model | FP | FN | Depth (mean diff / SD) (µm) | Arc (mean diff / SD) (º) | Thickness (mean diff /SD) (µm) | Depth ICC(2,1) | Calcium arc ICC(2,1) | Thickness ICC(2,1)
| ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- 
| RGB 2D | 9 | 10 | 1.09 ± 68.95 | -18.11 ± 21.97 | -47.64 ± 209.13 | 0.817 | 0.902 | 0.65 


## TODO:
 - Train pseudo 3d (+- 1, 2, 3 frames
 - Probability maps and uncertainty estimation: see losses functions and correlation with DICE
 - Model architecure and weights to see feature map (explainability)
 - Clean code and keep updating files
 - Add READMEs
 - Keep writing

