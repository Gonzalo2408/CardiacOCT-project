import SimpleITK
from pathlib import Path
import logging


logger = logging.getLogger("process_img")


def pre_process_image(
    input_image_path: Path, output_image_path: Path,
):
    """Method for pre-processing input images to be processed by nnunet

    - input_image_path: path to the original image file
    - output_image_path: path to the corresponding output
      pre-processed image file location (always .nii.gz format)

    Default behavior:
    check if output_image_path exists, if not
    read the original image and write it as an output image in .nii.gz format
    """
    if not output_image_path.is_file():
        img = SimpleITK.ReadImage(str(input_image_path))

        # Make sure that image datatype is not float
        if "float" in img.GetPixelIDTypeAsString():
            img = SimpleITK.Cast(img, SimpleITK.sitkInt16)

        SimpleITK.WriteImage(img, str(output_image_path))


def post_process_image(
    temp_results_dir: Path, output_dir: Path, overlay_destination: str
):
    """Method for post-processing the nnunet output

    - temp_results_dir: directory with artifacts generated by nnunet
    - output_dir: target output directory to write the post processed image to
    - overlay_destination: directory name (name only) in the output_dir to write the results to

    Default behavior:
    iterate over all the nnunet generated artifacts in the temp_results_dir and filter all .nii.gz files
    These segmentation files are subsequently written as .mha files in the
    {{output_dir}}/{{overlay_destination}} output folder.
    """
    overlay_output_dir = output_dir / overlay_destination
    overlay_output_dir.mkdir(exist_ok=True, parents=True)
    for seg_result in temp_results_dir.glob("*.nii.gz"):
        image_name = seg_result.name.split(".nii.gz")[0]
        logger.info(f"Writing segmentation mask for {image_name}...")
        img = SimpleITK.ReadImage(str(seg_result))
        mask_file = overlay_output_dir / f"{image_name}.mha"

        # Delete additional metadata that exists in nifti images but not mha images
        for k in img.GetMetaDataKeys():
            img.EraseMetaData(k)

        SimpleITK.WriteImage(img, str(mask_file), True)


def is_result_for_image_present(
    image_name: str, output_dir: Path, overlay_destination: str
) -> bool:
    """Method to check if all the output files for the image_name are present

    - image_name: filename without extension for the input scan
    - output_dir: target output directory to look for the output files
    - overlay_destination: directory name (name only) in the output_dir to where the results are stored

    Default behavior:
    Check for the existence of the {{output_dir}}/{{overlay_destination}}/{{image_name}}.mha file
    """
    overlay_output_dir = output_dir / overlay_destination
    result_image = overlay_output_dir / f"{image_name}.mha"
    return result_image.is_file()
