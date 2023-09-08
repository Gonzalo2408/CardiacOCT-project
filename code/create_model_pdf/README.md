## Model report

In this folder, a summary of a model prediction is generated as pdf. An overview can be seen in the figure below. The model report contains, for a given model and every frame in the test set:
 - OCT frame
 - Manual annotation
 - Model segmentation
 - Lipid arc and FCT measurements for the predicted segmentation
 - Calcium arc, thickness and depth for the predicted segmentation
 - Table that compare TCFA, calcium score, sidebranch, white and red thrombus, dissection and plaque rupture for the manual and predicted annotations.

<figure>
    <img src="/assets/model_pdf_example.png" alt="missing" />
    <figcaption>
        <strong>Figure 1.</strong> Example of model report for a frame for k = 3 model.
    </figcaption>
</figure>

Apart from this, the option to obtain an comparative of every model is also possible. Instead, each page in the PDF will contain the raw OCT frame, the manual segmentation, and the prediction by every model. An example on this can be seen below.


<figure>
    <img src="/assets/model_pdf_overview.png" alt="missing" />
    <figcaption>
        <strong>Figure 2.</strong> Example of model report for comparative of every model.
    </figcaption>
</figure>