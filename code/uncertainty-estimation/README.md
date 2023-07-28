## Uncertainty estimation

Usually, neural networks are too overconfident in their results, meaning that the probability to belong to a label that the model asigns to each pixel does not always relate to the true accuracy of that label. That is why we are building some uncertainty metrics in order to see how well calibrated is our model and give more insight to clinicians. 

### Feature maps

For explainability purposes, the feature maps after each convolutional layer are obtained using the state dictionary and the model architecture. (...)

### Probability maps

The nnUNet allows to store the probability maps for each image. That is, for each pixel, we get the softmax probabilities for each label (note that these probabilities sum 1). While these already give some information on how certain the model is about a prediction, it is usually overconfident. However, we also retrieved them as you can see in the corresponding Jupyter notebook. It basically plots like a "heatmap", so the areas with more clear softmax probability are yellow, and those areas with more uncertain softmax probabilities are blue.

![Figure 1. Example of probability map as overlay and corresponding image. It is worth noting that edges are always uncertain regions, but other tiny blue regions can be seen as well](/assets/prob_map.png)

The probability maps were analysed for each label (...)


### Calibration

Metrics such as the Expected Calibration Error (ECE) can give some information about how well the given probabilities represent the true accuracy of the model. For this project, the ECE was estimated. 
