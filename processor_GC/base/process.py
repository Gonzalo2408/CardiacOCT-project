import logging
import subprocess
import sys
import time
from shutil import rmtree
from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import os

import click

from process_img import (
    pre_process_image,
    post_process_image,
    is_result_for_image_present,
)


CONFIG_FILE = Path(__file__).parent / "config.json"

ALLOWED_EXTENSIONS = ("mha",)

logger = logging.getLogger("process")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--inputdir",
    "input_dir",
    default="/input",
    show_default=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    help="Directory path for input images",
)
@click.option(
    "--outputdir",
    "output_dir",
    default="/output",
    show_default=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    help="Directory path to store generated output (segmentation mask, etc.)",
)
@click.option(
    "--overlay-destination",
    default="images",
    show_default=True,
    help="Sub-directory of the output directory to store the obtained segmentation mask",
)
@click.option(
    "--task-id",
    default=17,
    show_default=True,
    type=click.INT,
    help="nnUNet task identifier",
)
@click.option(
    "--task-name",
    default="AbdominalOrganSegmentation",
    show_default=True,
    type=click.STRING,
    help="nnUNet task name (should match task identifier)",
)
@click.option(
    "--model",
    default="3d_fullres",
    show_default=True,
    type=click.STRING,
    help="nnUNet model identifier",
)
@click.option(
    "--use-test-time-augmentations",
    "--use-tta",
    "use_test_time_augmentations",
    default=False,
    show_default=True,
    type=click.BOOL,
    help="nnUNet inference uses test time augmentations (much slower, but usually better results)",
)
@click.option(
    "--use-overlap",
    "use_overlap",
    default=True,
    show_default=True,
    type=click.BOOL,
    help="nnUNet inference computes tiles with overlap during test time (slower, but more accurate)",
)
@click.option(
    "--folds",
    default=(),
    multiple=True,
    show_default=False,
    type=click.IntRange(min=0, max=None, clamp=True),
    help="Which nnUNet folds to use for prediction\n[default: all available]",
)
@click.option(
    "--num-threads-preprocessing",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, max=64, clamp=True),
    help="Number of threads to use for preprocessing",
)
@click.option(
    "--num-threads-nifti-save",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, max=64, clamp=True),
    help="Number of threads to use for saving nifti files",
)
@click.option(
    "--allowed-extensions",
    default=ALLOWED_EXTENSIONS,
    multiple=True,
    show_default=True,
    type=str,
    help="Supported extensions for input files.",
)
@click.option(
    "--write-json",
    default=False,
    show_default=True,
    type=click.BOOL,
    help="Write an output JSON file.",
)
@click.option(
    "--overwrite-output",
    default=True,
    show_default=True,
    type=click.BOOL,
    help="Set this to false to avoid recomputing already processed cases in the output_dir.",
)
def cli(
    input_dir: str,
    output_dir: str,
    overlay_destination: str,
    task_id: int,
    task_name: str,
    model: str,
    use_test_time_augmentations: bool,
    use_overlap: bool,
    folds: Optional[List[int]],
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    allowed_extensions: List[str],
    write_json: bool,
    overwrite_output: bool,
):
    process(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        overlay_destination=overlay_destination,
        task_id=task_id,
        task_name=task_name,
        model=model,
        use_test_time_augmentations=use_test_time_augmentations,
        use_overlap=use_overlap,
        folds=folds,
        num_threads_preprocessing=num_threads_preprocessing,
        num_threads_nifti_save=num_threads_nifti_save,
        allowed_extensions=set(allowed_extensions),
        write_output_json=write_json,
        overwrite_output=overwrite_output,
    )


def get_input_scans(
    input_dir: Union[Path, str], allowed_extensions: Set[str] = ALLOWED_EXTENSIONS
) -> Iterable[Path]:
    logger.info("Scans to process:")
    valid_input_scans = [
        filepath
        for filepath in Path(input_dir).iterdir()
        if (
            filepath.is_file()
            and any([filepath.name.endswith(ext) for ext in allowed_extensions])
        )
    ]
    logger.info("\n".join(map(str, valid_input_scans)))

    if len(valid_input_scans) == 0:
        msg = (
            f"No suitable images found in {input_dir}. Images should have one of the following extensions: "
            f"{', '.join(allowed_extensions)}."
        )
        raise InterruptedError(msg)
    else:
        yield from valid_input_scans


def get_image_name(file_name: Path, allowed_extensions: Set[str]) -> str:
    for ext in sorted(
        allowed_extensions, key=lambda v: -len(v)
    ):  # try to match longest extensions first
        dext = "." + ext
        if file_name.name.endswith(dext):
            return file_name.name[: -(len(dext))]
    raise ValueError(
        f"{file_name} did not have one of the allowed extensions: {allowed_extensions}"
    )


def prepare_image_for_nnunet(
    original_input_file: Path, nnunet_temp_input_dir: Path, allowed_extensions: Set[str]
):
    logger.info(f"Reading {original_input_file}")
    logger.info("Preparing image as _0000.nii.gz for nnunet...")
    image_name = get_image_name(
        file_name=Path(original_input_file), allowed_extensions=allowed_extensions
    )
    local_input_file = nnunet_temp_input_dir / f"{image_name}_0000.nii.gz"
    pre_process_image(
        input_image_path=original_input_file, output_image_path=local_input_file,
    )


def run_inference(
    task: str,
    nnunet_temp_input_dir: Path,
    nnunet_temp_output_dir: Path,
    model: str = "3d_fullres",
    use_test_time_augmentations: bool = False,
    use_overlap: bool = True,
    folds: Optional[List[int]] = None,
    num_threads_preprocessing: int = 1,
    num_threads_nifti_save: int = 1,
):
    # Run inference with nnUnet 3D full resolution model
    logger.info(f"Running nnUNet inference for {task}")

    cmd = [
        "nnUNet_predict",
        "-t",
        task,
        "-i",
        str(nnunet_temp_input_dir),
        "-o",
        str(nnunet_temp_output_dir),
        "-m",
        model,
        "--num_threads_preprocessing",
        str(num_threads_preprocessing),
        "--num_threads_nifti_save",
        str(num_threads_nifti_save),
    ]
    if not use_test_time_augmentations:
        cmd.extend(["--disable_tta"])
    if not use_overlap:
        cmd.extend(["--step_size", "1"])
    if folds is not None:
        folds = list(folds)
        if len(folds) > 0:
            cmd.extend(["--folds"] + [str(fold) for fold in folds])
    try:
        subprocess.run(cmd, encoding="utf-8", check=True)
        logger.info(f"Finished nnUNet inference for {task}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Segmentation failed") from e


def clean_nnunet_folders(nnunet_temp_input_dir: Path, nnunet_temp_output_dir: Path):
    def delete_folder_contents(folder):
        for path in Path(folder).iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)

    delete_folder_contents(nnunet_temp_input_dir)
    delete_folder_contents(nnunet_temp_output_dir)


def write_json_output(input_dir: Path, output_dir: Path, overlay_destination: str):
    logger.info("Writing json output...\n")
    mask_files = list((output_dir / overlay_destination).iterdir())
    image_names = [
        filepath.name for filepath in mask_files if (input_dir / filepath).is_file()
    ]

    # Create results.json file
    json_results_filepath = output_dir / "results.json"
    with open(json_results_filepath, "w") as f:
        json.dump(
            [
                {
                    "entity": image_name,
                    "metrics": {"segmentation": f"filepath:{mask_file}"},
                    "error_messages": [],
                }
                for image_name, mask_file in dict(zip(image_names, mask_files)).items()
            ],
            f,
        )
    logger.info("Finished!")


def read_defaults_from_config(config_file: Path = CONFIG_FILE) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid configuration file found at: {config_file}, not a dict JSON object."
        )
    return data


def process(
    input_dir: Path,
    output_dir: Path,
    overlay_destination: str,
    task_id: int,
    task_name: str,
    model: str,
    use_overlap: bool,
    use_test_time_augmentations: bool,
    folds: Optional[List[int]],
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    allowed_extensions: Set[str],
    write_output_json: bool,
    overwrite_output: bool,
):
    nnunet_temp_input_dir = Path("/tmp/input")
    nnunet_temp_output_dir = Path("/tmp/output")

    task = (
        f"Task{task_id:03d}_{task_name}"  # e.g. "Task017_AbdominalOrganSegmentation",
    )

    # Get list of valid input scans from folder
    input_scans = list(
        get_input_scans(input_dir=input_dir, allowed_extensions=allowed_extensions)
    )

    # Predict segmentations one scan at a time
    for idx, input_scan in enumerate(input_scans):
        if not overwrite_output and is_result_for_image_present(
            image_name=get_image_name(
                file_name=input_scan, allowed_extensions=allowed_extensions
            ),
            output_dir=output_dir,
            overlay_destination=overlay_destination,
        ):
            logger.info(
                f"{idx+1:4}/{len(input_scans):4} skipping input scan (results present): {input_scan}"
            )
            continue
        start_time = time.time()
        logger.info(
            f"{idx+1:4}/{len(input_scans):4} processing input scan: {input_scan}"
        )
        prepare_image_for_nnunet(
            original_input_file=input_scan,
            nnunet_temp_input_dir=nnunet_temp_input_dir,
            allowed_extensions=allowed_extensions,
        )
        run_inference(
            task=task,
            nnunet_temp_input_dir=nnunet_temp_input_dir,
            nnunet_temp_output_dir=nnunet_temp_output_dir,
            model=model,
            use_overlap=use_overlap,
            use_test_time_augmentations=use_test_time_augmentations,
            folds=folds,
            num_threads_nifti_save=num_threads_nifti_save,
            num_threads_preprocessing=num_threads_preprocessing,
        )
        post_process_image(
            input_image_path=input_scan,
            temp_results_dir=nnunet_temp_output_dir,
            output_dir=output_dir,
            overlay_destination=overlay_destination,
        )
        clean_nnunet_folders(
            nnunet_temp_input_dir=nnunet_temp_input_dir,
            nnunet_temp_output_dir=nnunet_temp_output_dir,
        )
        logger.info(
            f"Finished processing input scan: {input_scan} in {int(time.time() - start_time)} seconds"
        )
    if write_output_json:
        write_json_output(
            input_dir=input_dir,
            output_dir=output_dir,
            overlay_destination=overlay_destination,
        )
    logger.info("Finished processing...")


def setup_logging(log_level: int = logging.DEBUG,):
    log = logging.getLogger("")
    log.setLevel(log_level)
    fmt = logging.Formatter(
        "%(asctime)s [%(name)-11s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    log.addHandler(ch)


def main():
    setup_logging()
    defaults = read_defaults_from_config(config_file=CONFIG_FILE)
    cli(default_map=defaults)


if __name__ == "__main__":
    main()
