## Uncertainty estimation

Usually, neural networks are too overconfident in their results, meaning that the probability to belong to a label that the model asigns to each pixel does not always relate to the true accuracy of that label. That is why we are building some uncertainty metrics in order to see how well calibrated is our model and give more insight to clinicians. 

### Feature maps

For explainability purposes, the feature maps after each convolutional layer are obtained using the state dictionary and the model architecture. (...)

### Probability maps

The nnUNet allows to store the probability maps for each image. That is, for each pixel, we get the softmax probabilities for each label (note that these probabilities sum 1). An example of this map can be seen below, which looks like a heatmap with the most confident areas in yellow and least confident in blue. We analyzed these probability maps from two perspectives:

- By calculating the total Expected Calibration Error (ECE) in the test set.
- By obtaining the confidence on lipid and calcium regions, considering if each frame is TP, TN, FP or FN. 

![Figure 1. Example of probability map as overlay and corresponding image. It is worth noting that edges are always uncertain regions, but other tiny blue regions can be seen as well](/assets/prob_map.png)

The result on this aspect can be found [here](info-files/uncertainty/).