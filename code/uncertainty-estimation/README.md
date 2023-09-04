## Uncertainty estimation

Usually, neural networks are too overconfident in their results, meaning that the probability to belong to a label that the model asigns to each pixel does not always relate to the true accuracy of that label. That is why we are building some uncertainty metrics in order to see how well calibrated is our model and give more insight to clinicians. 

### Optimal lipid and calcium threshold

The lipid and calcium thresholds are determined using the notebook in **get_lipid_cal_thresh.ipyn**. Here, the ROC is first computed using the train predictions on the k = 3 model. Then, an optimal value of 1700 and 100 pixels is found for lipid and calcium regions, respectively. Finally, the threshold is applied to the test set predictions and the sensitivity and specificity are again computed.

### Feature maps

For XAI, the feature maps were retrieved from a couple of frames and for the 2D and k = 3 models. This was done my modifying the base nnU-Net forward function so the tensor after the convolutional layers are saved as .pt files. The features are retrieved from the 8 layers in the encoder, and for each layer, 5 features corresponding to the 5 folds are obtained.

The **feature_maps.ipynb** notebook processes these .pt files. First, for each feature map, an average of the 5 folds is obtained. Then, the maximum value in each pixel is obtained so we compress all the 3D complexity to a 2D image. Each feature map is then saved in the PNG format and further visually analysed.

### Probability maps

The nnUNet allows to store the probability maps for each image. That is, for each pixel, we get the softmax probabilities for each label (note that these probabilities sum 1). An example of this map can be seen below, which looks like a heatmap with the most confident areas in yellow and least confident in blue. We analyzed these probability maps from three perspectives:

- Expected Calibration Error (ECE) and reliability curves: the ECE was obtained for the total of the test set to analyze if the trained models are over or underconfident on their predictions. For this, we also plot the reliability curves, with the total confidences on the x-axis, and the accuracy on the y-axis.

- Total confidence on lipid and calcium regions: based on the idea that each frame can be TP, TN, FP and FN on predicting lipid or calcium, we studied the total confidence for each case. In the case of TP and FP, we average the confidence of the predicted lipid/calcium region. On the other hand, for FN we average the confidence of the region in the predicted segmentation that contains the lipid/calcium in the original segmentation. Finally, for TN, we do not compute any confidence values.

- Entropy and DICE: finally, we compute the entropy for each pixel in a frame prediction, using the probability maps. For each model and frame, we obtain the average entropy on lipid and calcium regions on TP cases, since FP and FN give a DICE of 0. After this, the DICE and entropy for TP in lipid and calcium correlation is obtained for each model, plus the average entropy. 

<!-- ![Figure 1. Example of probability map as overlay and corresponding image. It is worth noting that edges are always uncertain regions, but other tiny blue regions can be seen as well](/assets/prob_map.png) -->

<figure>
    <img src="/assets/prob_map.png" alt="missing" s/>
    <figcaption>
        <strong>Figure 1.</strong> Example of probability map as overlay and corresponding image. It is worth noting that edges are always uncertain regions, but other tiny blue regions can be seen as well
    </figcaption>
</figure>

The result on this aspect can be found [here](/info-files/uncertainty/).