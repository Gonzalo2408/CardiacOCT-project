## Metrics file

The file in this folder contains all the generated metrics. Below, an explanation on every sheet that can be found.

### Overview test

This sheet contain the summary of every label in the study, except for the arc and TCFA metrics. For every model, you will find the average metrics for DICE per frame and per pullback, PPV, NPV, senstivity, specificity andf Cohen's Kappa, for every label. Moreover, a plot comparing the DICEs per frame can also be found, as can be seen below.

![Figure 1. DICEs per frame for every model](/assets/dices_per_frame.png)


### TCFAs

In this sheet, you will find the PPV, NPV, specificity, sensitivity and Kappa for the detected TCFAs, for every model.

### Lipid and calcium arc DICEs

Two sheets contain the results for the lipid arc and the calcium arc DICEs respectively. Each sheet shows, on the top, an overview table with the DICEs for a type of arc (both per frame and per pullback), with a bar plot to the right to better visualize each model results. These plots can be visualized below. 

<p float="left">
  <img src="/assets/lipid_arc_dices.png" width="500" height=300"/>
  <img src="/assets/cal_arc_dices.png" width="500" height="300"/> 
</p>


Finally, on bottom left, you will find a table with the DICE for every frame and every model, and to the bottom right the arc DICE for every pullback.


### DICEs sheets

Every model has its own sheet for showing the DICEs. On each sheet, you will first see a table with the DICEs for every frame and every label. Below that table, you will find another table with the DICE results on pullback level for that model.