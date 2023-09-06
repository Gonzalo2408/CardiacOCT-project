## Statistics and measurements

There are two files in this folder: the measures file, which contains all the measurements of lipid and calcium for every model, with the manual measurements, and the measures analysis file, which contains all the statistical results on the measured regions (lipid arc, FCT, calcium arc, calcium thickness and calcium depth).

### Measures file

This file is manually created using the measurments obtained in the [counts](/info_files/counts) file. The automated measurements were obtained from the annotator and copied in this file as well. 

### Measures analysis file

Each sheet in this file corresponds to one of the 5 measurements we are analyzing, plus an extra sheet contains different plots to compare each model. Below, you can find a plot example for the ICC.

<p float="left">
  <img src="/assets/icc_lipid.png" width="500" />
  <img src="/assets/icc_cal.png" width="500" /> 
  <figcaption>
        <strong>Figure 1.</strong> ICC evolution for lipid and calcium measurements.
    </figcaption>
</p>


For a given sheet, each row corresponds to one model, containing the nº of FP and FN, mean difference, standard deviation, correlation, ICC(2,1), nº of outliers and a list containing all the specific frames that are outliers. These values are the one used then for the plots.