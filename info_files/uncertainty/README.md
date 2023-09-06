## Uncertainty file

The results on the confidence values can be seen in the Excel file in this folder. Below, an explanation on each sheet.

### Plots

The first sheet contains an overview on the confidence results. For both lipid and calcium, you will find a table with each row containing the specific model and the average confidence for FP, FN and TP. To the right of each table, you will see a bar plot representing the difference between each case.

<p float="left">
  <img src="/assets/lipid_confs.png" width="500" />
  <img src="/assets/calcium_confs.png" width="500" /> 
  <figcaption>
        <strong>Figure 1.</strong> Lipid and calcium average confidences, depending if they are TP, FP or FN, for every model.
  </figcaption>
</p>


### Confidence sheets

The second and third sheets correspond to the confidence values for lipid and calcium, respectively. You will find a table with every frame and, for that frame, if it is FP, FN, TP or TN, and the confidence associated, for every model (note that for TN, there will no be confidence values).

### Reliability curves

The reliability curves, with the total ECE in the test set, can be found below. As we can see, the ECE on every trained model is very close to 0, meaning that the nnU-Net is very well calibrated. The reliability curves also reafirm this idea.

<p float="left">
  <img src="/assets/ece_conf_model_2d.png" width="500" />
  <img src="/assets/ece_conf_pseudo3d_1.png" width="500" /> 
  <img src="/assets/ece_conf_pseudo3d_2.png" width="500" />
  <img src="/assets/ece_conf_pseudo3d_3.png" width="500" />
  <figcaption>
        <strong>Figure 2.</strong> Reliability curves and ECE for model 2D (top left), k = 1 (top right), k = 2 (bottom left) and  k = 3 (bottom right)
  </figcaption>
</p>