## Pseudo 3D dataset conversion

With this conversion, we aim to solve the problems with the 3D training while also keeping features from the nature of this training (i.e spatial features). 

As explained in the introduction, we sampled the k neighbours before and after each annotated framed for the training. Essentially, we can use this neighbour frames as new modalities for every annotated frame. We basically keep the annotated frame and sample k frames before and after. If one of this neighbours is out of range, we just use a empty array, so we always have every annotated frame in the middle of the generated volume.

Note that we only perform this operation for the images. We keep the same segmentations, that is, each annotated frame in separated files. 
