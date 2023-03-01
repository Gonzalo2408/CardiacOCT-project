# Helper for building nnU-net processor docker images

This is a tool for bundling trained nnU-net models in a docker image that runs the model on new images, both on SOL and as grand-challenge.org algorithm.
Next to using self-trained models, pre-trained models that can be downloaded from zenodo are also supported.

Note that running multiple network types (2D/3D/3D cascades) is currently not supported.

## Usage

Create and/or navigate to an empty directory on a disk with sufficient space, e.g.:
   
```bash
mkdir /tmp/your-name/build-processor && cd /tmp/your-name/build-processor
```

Make sure that you have a copy of the `/processor` folder somewhere as this contains the scripts needed to build your docker image.

### Configuration

Create and configure a JSON file called `config.json`. This JSON file should contain all the default values for calling your nnUNet processor. For example, for a pretrained hippocampus segmentation model this looks like:

```json
{
  "task_id": 4,
  "task_name": "Hippocampus",
  "model": "3d_fullres",
  "use_test_time_augmentations": false,
  "use_overlap": true,
  "folds": null,
  "overlay_destination": "images",
  "num_threads_preprocessing": 1,
  "num_threads_nifti_save": 1,
  "allowed_extensions": ["mha", "mhd", "nii.gz"],
  "write_json": false
}
```

The value of `overlay_destination` depends on the interface defined on grand-challenge.org for the output and is relative to `/output`. The
default value of "images" corresponds to the generic overlay output.

### Building the docker image

Run the `build_processor.py` Python script (stand-alone compatible with SOL machines Python 3.8) to create your docker processor.

If you already have a folder with existing/trained model weights:
```bash
./build_processor.py --config-file /path/to/config.json --model-dir /path/to/trained/model/weights
```

Otherwise, if you want to download available pre-trained weights:
```bash
./build_processor.py --config-file /path/to/config.json --download-pretrained
```

Optionally, you can specify the `--tag` flag to create a custom docker image tag for the build. By default, this is: `doduo1.umcn.nl/nnunet/processor-task{task_id}-{task_name}:{timestamp}`.
Also, you can specify the `--keep-build-files` flag to retain all the build files (build context) for speed-up for consecutive builds. By default, these are removed, so if you use the flag you are responsible for cleaning the build files once you are finished!

Specify `--push` to also automatically push the image to a remote repository and `--export` to save the image to a `.tar.gz` file that can be uploaded to grand-challenge.org

For custom image pre- and post-processing you can replace the `process_img.py` file by specifying a path to an 
alternative python file using the `--process-img-file` option. Have a look at the default `processor/base/process_img.py` 
file to find out how to implement your own methods.

### Frozen or killed processor

nnU-net can use a lot of memory (RAM), especially if images are large or if there are a large number of classes. If you run into memory
issues using a normal processor build, you can apply the experimental `--apply-resample-patch` option, which applies a patch for nnunet that
optimizes the memory usage in the step where the probability tensor is resampled from the working resolution to the original resolution.
Note that this patch is experimental and might cause unintended issues.

## Base image
The minimal processor docker image relies on a base image that should always be available from the SOL docker repository.
If this is not the case, you can build and push the base image to the repository by running:

```shell
./build_base_image.sh
```
