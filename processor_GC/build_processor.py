#!/usr/bin/python

import argparse
import time
from pathlib import Path
import subprocess as sp
from typing import Optional, Dict
import json
import shutil
import os
import re


SCRIPT_DIR = Path(__file__).parent
DOCKER_EXECUTABLE = "docker"
with open(SCRIPT_DIR / "base" / "requirements.txt") as requirements_file:
    REQUIREMENTS_DATA = requirements_file.read()
CUDA_VERSION = re.search(r"\+cu(\d+)", REQUIREMENTS_DATA).groups()[0]
TORCH_VERSION = re.search(r"torch==(\d+\.\d+\.\d+)", REQUIREMENTS_DATA).groups()[0]
NNUNET_VERSION = re.search(r"nnUNet\.git@(\d+\.\d+\.\d+-\d+)", REQUIREMENTS_DATA).groups()[0]
TEMPLATES_DIR = SCRIPT_DIR / "templates"
DEFAULT_PROCESS_IMG_FILE = SCRIPT_DIR / "base" / "process_img.py"
DEFAULT_BASE_IMAGE = f"doduo1.umcn.nl/nnunet/processor-base:cu{CUDA_VERSION}-pt{TORCH_VERSION}-nnunet{NNUNET_VERSION}"
DEFAULT_DOCKER_TAG = (
    "doduo1.umcn.nl/nnunet/processor-task{task_id}-{task_name}:{timestamp}"
)
DEFAULT_BUILD_DIR = Path(os.getcwd()) / "build"


def download_pretrained_weights(download_dir: Path, task_id: int, task_name: str):
    error_msg = "download_pretrained_weights - Docker command: {cmd} did not complete successfully, aborting..."
    task = get_model_name(task_id=task_id, task_name=task_name)
    print(f"attempting to download pretrained model files for {task}...")
    container_name = f"nnunet_downloader_{int(time.time())}"
    cmd = [
        str(DOCKER_EXECUTABLE),
        "run",
        "--name",
        container_name,
        "--entrypoint",
        "nnUNet_download_pretrained_model",
        "--env",
        "RESULTS_FOLDER=/tmp/model",
        DEFAULT_BASE_IMAGE,
        task,
    ]
    try:
        result = sp.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                error_msg.format(cmd=cmd)
            )

        # Copy downloaded data
        download_dir.mkdir(exist_ok=True, parents=True)
        cmd = [
            str(DOCKER_EXECUTABLE),
            "cp",
            f"{container_name}:/tmp/model/.",
            str(download_dir)
        ]
        result = sp.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                error_msg.format(cmd=cmd)
            )
    finally:
        # Clean up
        print(f"Attempt to cleanup temporary docker container: {container_name}")
        sp.run([
            str(DOCKER_EXECUTABLE),
            "rm",
            container_name
        ])


def verify_config_file(config_file: Path) -> Dict:
    if not config_file.is_file():
        raise RuntimeError(f"No config file found at: {config_file}")
    try:
        with open(config_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load the configuration file with error: {e}")


def copy_file(src_file: Path, target_path: Path):
    full_src_file_path = src_file.resolve().absolute()
    target_path = target_path.resolve().absolute()
    if target_path.is_dir():
        target_path = target_path / src_file.name
    if full_src_file_path != target_path:
        print(f"Copying file {full_src_file_path} -> {target_path}")
        shutil.copyfile(str(full_src_file_path), str(target_path))


def copy_install_files(target_dir: Path):
    for src_file in [
        TEMPLATES_DIR / "Dockerfile",
        SCRIPT_DIR / "base" / "process.py"
    ]:
        copy_file(src_file=src_file, target_path=target_dir)


def get_model_name(task_id: int, task_name: str) -> str:
    return (
        f"Task{task_id:03d}_{task_name}"  # e.g. "Task017_AbdominalOrganSegmentation",
    )

def copy_fonts(src_dir: Path, target_dir: Path):
    shutil.copytree(
        str(src_dir),
        str(target_dir / 'fonts'),
        dirs_exist_ok=True,
    )

def copy_trained_weights(src_dir: Path, target_dir: Path, task_id: int, task_name: str):
    if src_dir.resolve().absolute() == target_dir.resolve().absolute():
        print(
            f"source and target model dirs ({src_dir}) are the same, no model files were copied..."
        )
    else:
        model_name = get_model_name(task_id=task_id, task_name=task_name)

        def filtered_copyfile(src: str, dest: str):
            if model_name in src and not src.endswith(".nii.gz") and not Path(dest).is_file():
                print(f"Copying {src}")
                shutil.copyfile(src, dest)

        shutil.copytree(
            str(src_dir),
            str(target_dir),
            copy_function=filtered_copyfile,
            dirs_exist_ok=True,
        )


def complete_docker_file(
    docker_file: Path,
    base_image: str = DEFAULT_BASE_IMAGE,
    apply_resample_patch: bool = False
):
    print(f"Writing Dockerfile {docker_file} with base image {base_image} "
          f"and apply resample patch {apply_resample_patch}")
    with open(docker_file, "r") as f:
        data = f.read()
    data = data.replace("{{baseimage}}", base_image)
    data = data.replace("{{DIAG_NNUNET_ALT_RESAMPLING}}", ("1" if apply_resample_patch else "0"))
    with open(docker_file, "w") as f:
        f.write(data)


def build_processor(
    config_file: Path,
    build_dir: Path,
    model_dir: Optional[Path] = None,
    fonts_dir: Optional[Path] = None,
    download_pretrained: bool = False,
    tag_name: str = DEFAULT_DOCKER_TAG,
    keep_build_files: bool = False,
    base_image: str = DEFAULT_BASE_IMAGE,
    push_image: bool = False,
    export_image: bool = False,
    process_img_file: Path = DEFAULT_PROCESS_IMG_FILE,
    apply_resample_patch: bool = False,
):
    build_dir.mkdir(exist_ok=True)
    cfg = verify_config_file(config_file=config_file)
    task_id = cfg["task_id"]
    task_name = cfg["task_name"]
    print(f"Found the following configuration file {config_file} with settings:\n{cfg}")
    copy_file(src_file=config_file, target_path=build_dir / "config.json")
    copy_install_files(target_dir=build_dir)
    copy_file(src_file=process_img_file, target_path=build_dir / "process_img.py")
    copy_fonts(src_dir=fonts_dir, target_dir=build_dir)

    complete_docker_file(
        docker_file=build_dir / "Dockerfile",
        base_image=base_image,
        apply_resample_patch=apply_resample_patch
    )

    build_model_dir = build_dir / "models"
    build_model_dir.mkdir(exist_ok=True)
    if download_pretrained:
        download_pretrained_weights(
            download_dir=build_model_dir, task_id=task_id, task_name=task_name
        )
    elif model_dir is not None:
        copy_trained_weights(
            src_dir=model_dir,
            target_dir=build_model_dir,
            task_id=task_id,
            task_name=task_name,
        )
    else:
        print("WARNING! no model weights will be embedded in the processor, "
              "did you forget to set the --model-dir or --download-pretrained flags?")

    docker_tag = tag_name.format(
        task_id=task_id, task_name=task_name.lower(), timestamp=int(time.time())
    )
    print(f"Attempting to build docker image with tag: {docker_tag} in {build_dir}")
    cmd = [
        str(DOCKER_EXECUTABLE),
        "build",
        "--tag",
        docker_tag,
        "--target",
        "processor",
        str(build_dir),
    ]
    result = sp.run(cmd, env={"DOCKER_BUILDKIT": "1"},)
    if result.returncode != 0:
        raise RuntimeError(
            f"ERROR: Docker command: {cmd} did not complete successfully, aborting..."
        )
    if keep_build_files:
        print(
            f"keep_build_files was specified so the files under: '{build_dir}' will be retained. "
            f"Don't forget to remove these when you are done!"
        )
    else:
        print(f"Removing all build files under: '{build_dir}'")
        shutil.rmtree(str(build_dir))

    # Push and/or export docker image
    if push_image:
        print(f"Pushing docker image with tag {docker_tag} to the remote repository")
        cmd = [
            str(DOCKER_EXECUTABLE),
            "push",
            docker_tag,
        ]
        result = sp.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: Docker command: {cmd} did not complete successfully")
            return

    if export_image:
        image_file = docker_tag.split("/", maxsplit=1)[-1].replace("/", "-").replace(":", "-") + ".tar.gz"
        print(f"Exporting docker image to {image_file}")

        with open(image_file, "wb") as fp:
            with sp.Popen((str(DOCKER_EXECUTABLE), "save", docker_tag), stdout=sp.PIPE) as ps:
                result = sp.run(("gzip", "-c"), stdin=ps.stdout, stdout=fp)

        if result.returncode != 0:
            print(f"ERROR: Exporting the docker image {docker_tag} did not complete successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Build a SOL compliant nnUNet processor using existing trained models and a configuration file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_DOCKER_TAG,
        help=f"Docker tag to assign to the build image. default format is {DEFAULT_DOCKER_TAG}",
    )
    parser.add_argument(
        "--push",
        action='store_true',
        help="Pushes the docker image to the remote repository when the build was successful",
    )
    parser.add_argument(
        "--export",
        action='store_true',
        help="Exports the docker image into a .tar.gz file when the build was successful",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(DEFAULT_BUILD_DIR),
        help="Path to build directory. default: 'build' directory in current working dir",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(Path(os.getcwd()) / "config.json"),
        help="Path to configuration file, default looks for 'config.json' in current working dir",
    )
    parser.add_argument(
        "--process-img-file",
        type=str,
        default=str(DEFAULT_PROCESS_IMG_FILE),
        help="Path to process_img.py file defining the image processing before and after running nnunet predict. "
        "override the default file to use your own image processing file."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(os.getcwd()) / "models"),
        help="Path to model directory, default looks for 'models' folder in current working dir",
    )
    parser.add_argument(
        "--fonts-dir",
        type=str,
        default=str(Path(os.getcwd()) / "fonts"),
        help="Path to model directory, default looks for 'models' folder in current working dir",
    )
    parser.add_argument(
        "--download-pretrained",
        action="store_true",
        help="Set this if your nnUNet task model weights can be downloaded remotely, look at the "
        "following link for a list of all supported models: "
        "https://github.com/MIC-DKFZ/nnUNet/blob/f579199deefb8f1ecf20bb000d5e8b559015e578"
        "/nnunet/inference/pretrained_models/download_pretrained_model.py#L23",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=DEFAULT_BASE_IMAGE,
        help=f"Set the base image to overwrite the default base image. Default: {DEFAULT_BASE_IMAGE}",
    )
    parser.add_argument(
        "--keep-build-files",
        action="store_true",
        help="Set this flag if you want to prevent removal of the build files at the end of the build process, "
        "this will increase the speed of consecutive builds and allows to copy the build artifacts."
        "You'll have to remove the build-dir manually if you specify this.",
    )
    parser.add_argument(
        "--apply-resample-patch",
        action="store_true",
        help="Set this flag if you want to apply the experimental resampling nnunet patch to reduce "
             "the memory footprint of nnunet prediction. WARNING use at your own risk, might cause bugs."
    )
    args = parser.parse_args()

    build_processor(
        build_dir=Path(args.build_dir),
        config_file=Path(args.config_file),
        model_dir=Path(args.model_dir),
        fonts_dir = Path(args.fonts_dir),
        download_pretrained=args.download_pretrained,
        tag_name=args.tag,
        keep_build_files=args.keep_build_files,
        base_image=args.base_image,
        push_image=args.push,
        export_image=args.export,
        process_img_file=Path(args.process_img_file),
        apply_resample_patch=args.apply_resample_patch,
    )


if __name__ == "__main__":
    main()
