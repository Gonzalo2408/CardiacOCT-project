## Statistical analysis

The statistical analyis on lipid (arc and FCT) and calcium (arc, thickness and depth) is performed with the aim to compare manual and automated measurements on these regions. The functions that perfom this analysis are in the class stored in **get_statistical_analysis.py**, which is compiled with the Shell file in this folder. Before running this analysis, you should have built an Excel file containing the measurements for both manual and automated segmentations. In this case, the file I'm using is called measures_V2, which you cand find [here](/info_files/statistics/). The Shell file contains the following arguments:

- model_id: it should be the exact model name that you are in your measures Excel file for naming all the columns. So, for example, if your column name is "FCT/Lipid arc pseudo 3D 1", the model_id should be "pseudo 3D 1".

- output_filename: name of the Excel file you want to generate with the statistics results

- png_path: path to the folder in which you want to store the Bland-Altman and correlation plots.


The statistical analysis includes: 

 - False Positive and False Negatives: computed in the **get_data_filtered** function, which also removes FP, FN, nulls and NaNs for performing a fair analysis.
 - Mean difference and standard deviation: computed in **mean_sd**.
 - Correlation + plot: the correlation is obtained with the **corr** function, and the plot is saved with **scatter_data_save_png**.
 - Bland-Altman plots: saved with the function **save_bland_altman**.
 - Intra-class correlation (ICC(2,1)): it is calculated with the **calculate_icc** function.
 - Outliers: using either Z-score or Tukey approaches, which are found using the **find_outliers** function