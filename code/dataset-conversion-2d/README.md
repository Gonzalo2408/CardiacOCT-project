## 2D dataset conversion

In this folder, the original dataset is converted into 2D slices for the 2D training with the mentioned preprocessing steps (resampling + circular mask). Morevoer, each slice is saved as a pseudo-volume by adding an extra empty dimension. Further sanity checks for NaN values or new weird labels are also included.

For the images conversion, this is submitted to the SOL cluster using a Docker container since this takes longer to convert (~2 hours for the latest dataset). 


![](../../../Images/2d_dataset_conversion.png)

### Additional scripts

The scripts such as generate_json_dataset.py or file_and_folder_operations.py are only used to create the .json file with the dataset information in a fast way. This .json file is needed for each training for the nnUNet.