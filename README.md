# CardiacOCT project

## Project description

Acute myocardial infarction (MI) remains as one of the leading causes of mortality in the developed world. Despite huge advances in treating this condition such as the use of angiograms to locate the occluded artery or coronary angioplasty, there is still a debate on whether to treat certain lesions found during coronary angiography or not, since it is difficult to predict which plaques would have a worse outcome for the patient’s health. Imaging modalities such as intracoronary optical coherence tomography (OCT) provide a better comprehension of plaque characteristics and it can help surgeons to better asses these lesions, improving patient’s outcome.

In this project, an automatic segmentation model will be designed for intracoronary OCT scans in order to asses for plaque vulnerability and detect other abnormalities such as white or red thrombus or plaque rupture. Specifically, a nnUNet (no-new-UNet) that works with sparse annotated data will be designed. Initially, the model will be trained on singles frames that contain a corresponding segmentation map, that is, the model works in a supervised manner. Next, in order to account for the sparse annotations, a 3D UNet will be trained in a semi-supervised manner. After the models have been trained, several automatic post-processing techninques for lipid arc and cap thickness measurement will be implemented. Moreover, an uncertainty estimation model will be designed in order to detect unreliable segmentations and add more value to the algorithm's output.

## Dataset

The intracoronary OCT dataset used in this study is a collection of OCT scans from different centers, including (write it better later) RadboudUMC, EST-NEMC, AMPH, HMC, ISALA.

Since manually labelling the dataset is a very time consuming task for annotators, not all scans were included for the training. In particular, the methodology is to label each 40th frame in the scan, unless there are some regions in other frames that are necessarily to label. Thus, frames that contain some degree of labelling were included for the training. 

| Dataset  | Nº of patients (train/test) | Nº of pullbacks (train/test) | Nº of annotated frames (train/test)
| ------------- | ------------- | -------------  | -------------
| First dataset  | 49/13 (1 EST-NEMC, 24 AMPH, 3 HMC, 24 ISALA, 10 RADB)  | 56/14  | 783/163
| Second dataset  | 75/13 (1 EST-NEMC, 27 AMPH, 3 HMC, 24 ISALA, 33 RADB)  | 88/14  | 1215/162 


The ROIs for each OCT scan are (note that for each dataset, only the train set is shown):

| ROIs  | Distribution first dataset (frames/pullbacks)(%) | Distribution second dataset (%) | Test set (%)
| ------------- | ------------- | ------------- | -------------  
| Lumen  | - | - | -
| Guidewire  | - | - | -
| Wall | - | - | -
| Lipid | 51 / 98 | 47 / 98 | 48 / 93
| Calcium | 27.58 / 83.92 | 27.07 / 81.81 | 16.67 / 71.43
| Media | 95.89 / 100 | 96.21 / 100 | 99.38 / 100
| Catheter | - | - | -
| Sidebranch | 13.79 / 6.89 | 14.97 / 89.77 | 16.67 / 71.42
| Red thrombus | 6.89 / 26.78 | 5.67 / 23.86 | 0.61 / 7.14
| White thrombus | 5.61 / 28.57 | 4.53 / 23.86 | 0 / 0
| Dissection | 0.76 / 5.35 | 0.49 / 3.41 | 0 / 0
| Plaque rupture | 7.02 / 25 | 5.59 / 21.59 | 3.08 / 14.28

"-" indicates that the label is present in every pullback and in every frame of the dataset


## Preprocessing

The general preprocessing consisted of reshaping the images to a common size, which was (704, 704) and applying a circular mask to each slice. This is because each slice contains a watermark by Abbott with a small scale bar, and we do not our algorithm to learn from this information.

### 2D approach

For the 2D approach, the slices that did not contain any label were omitted. Thus, each slice for every pullback in the dataset was saved to a single NifTI file. In addition, each channel in the slice (RGB values) were saved separately as well, obtaining 3 files for each frame in the pullback. Similiary, each segmentation frame was saved in a different NifTI file. In this case, the segmentation is 1-dimensional, so there was no need to create a file for each dimension.

### 3D approach

For the 3D version of the nnUNet, a sparse trainer was used. In this case, the loss function is computed using slices that contain annotattions in each 3D volume. The frames that do not contain any label have a segmentation map that only contains -1, in order to the algorithm to detect unlabeled data. The preprocessing steps are very similar to the 2D model, in which each pullback is separated into its RGB values and each volume is saved separately in different NifTI files. Then, the main difference is that now whole 3D volumes are saved, rather than single 2d frames.


## Results

We obtained a bunch of metrics (accuracy, recall, jaccard, etc) but we only diplay the DICE scores for each one of the regions segmented



## TODO:
 - Train data with second dataset included (check case NLD-RADB-0030)
 - Build metrics Excel with both datasets results (including DICE results per region, plots, etc)
 - Solve NaN problem with 3d version of (first/second??) dataset --> impute -1 once preprocessing is done
 - Figure out post-processing techniques for lipid arc and cap thickness measurements (using dynammic programming + semantic segmentation of lipid, lumen, intima, etc). 
 For this, check [Lee et al. (2022)](https://www.nature.com/articles/s41598-022-24884-1) and [Wang et al. (2012)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3370980/)


