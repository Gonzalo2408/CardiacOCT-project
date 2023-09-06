## Dataset conversion

This folder contains the scripts that perform the dataset conversion to a format that can be recognized and manipulated by the nnUNet. We distinguish between two cases:

## 2D case

In this folder, the original dataset is converted into 2D slices for the 2D training. The preprocessing steps (resampling + circular mask) are performed in these files, for both DICOMs and segmentations files. 

There is a further script to perform the dataset conversion for the lipid training (i.e all labels that are not lipid are considered background and lipid is the foreground).

<!-- ![Figure 1. Preprocessing framework ](/assets/2d_dataset_conversion.png) -->

<p>
    <img src="/assets/2d_dataset_conversion.png" alt>
    <span style="font-style: normal;">
        <strong>Figure 1.</strong> Preprocessing framework
    </span>
</p>


## Pseudo 3D case

With this conversion, we aim to solve the problems with the 3D training while also keeping spatial in formation.

As explained in the introduction, we sampled the k neighbours before and after each annotated framed for the training. Essentially, we can use this neighbour frames as new modalities for every annotated frame. We basically keep the annotated frame and sample k frames before and after. If one of this neighbours is out of range, we just use a empty array, so we always have every annotated frame in the middle of the generated volume.