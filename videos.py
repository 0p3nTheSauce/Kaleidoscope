#!/usr/bin/env python3

import cv2
import os
import subprocess
import sys
from pathlib import Path
from cv2.typing import MatLike
from typing import List, Optional, Tuple
from argparse import ArgumentParser


def read_dir(path: Path, suff: str = ".jpg") -> List[MatLike]:
    """Read images (in order) from a directory.

    Args:
        path (Path): Path to image directory.
        suff (str, optional): Image filename suffix. Defaults to ".jpg".

    Returns:
        List[MatLike]: Images in the directory.
    """
    files = sorted(
        [f for f in path.iterdir() if f.is_file() and f.name.endswith(suff)],
        key=lambda f: int(f.stem),
    )
    imgs = []
    for f in files:
        imgs.append(cv2.imread(str(f), cv2.IMREAD_COLOR))

    return imgs


def images_to_video2(
    imgs: List[MatLike], out_path: str, fr: float = 30.0, video_code="mp4v"
) -> None:
    """Convert a list of images to a video (.mp4 produced not compatible with whatsApp).

    Args:
        imgs (List[MatLike]): _description_
        out_path (str): _description_
        fr (float, optional): _description_. Defaults to 30.0.
        video_code (str, optional): _description_. Defaults to "mp4v".

    Raises:
        RuntimeError: _description_
    """
    
    
    code_n_suff = {"XVID": ".avi", "MJPG": ".avi", "mp4v": ".mp4", "H264": ".mp4"}

    suffix = code_n_suff[video_code]
    if not out_path.endswith(suffix):
        out_path = out_path + suffix

    h, w, _ = imgs[0].shape
    size = (w, h)  # NOTE is different to regular opencv

    fourcc = cv2.VideoWriter_fourcc(*video_code)  # type: ignore (it doesn't like VideoWriter_fourcc)
    writer = cv2.VideoWriter(out_path, fourcc, fr, size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to create video writer with codec '{video_code}'. "
            f"Try 'MJPG' or 'mp4v' instead."
        )

    for img in imgs:
        writer.write(img)

    writer.release()
    print(f"Video saved at {out_path}")


def convert_mp4(name):
    # re-encode the .mp4

    original_file = f"{name}.mp4"
    temp_file = f"{name}_temp.mp4"

    # FFMPEG command to re-encode to H.264 and AAC
    command = [
        "ffmpeg",
        "-i",
        original_file,
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-strict",
        "-2",
        temp_file,
    ]

    try:
        subprocess.run(command, check=True)
        os.replace(temp_file, original_file)
        print(f"Video successfully written at {original_file}")
    except subprocess.CalledProcessError as e:
        print("An error occurred while processing the video:", e)
        # Clean up temp file if there was an error
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = ArgumentParser(description="Create a .mp4 video from a folder of images")

    parser.add_argument("directory", type=str, help="Image directory")

    parser.add_argument(
        "-sx", "--suffix", type=str, help="Image suffix.", default=".jpg"
    )

    parser.add_argument(
        "-ot",
        "--output",
        type=str,
        help="Output path, otherwise use name of directory",
        default=None,
    )
    
    
    args = parser.parse_args()

    path = Path(args.directory)

    if not path.exists():
        print(f"{path} not found")
        return
    if not path.is_dir():
        print(f"{path} is not a directory")
        return

    imgs = read_dir(path, suff=args.suffix)

    if len(imgs) == 0:
        print(f"no images found for directory: {path} with image suffix: {args.suffix}")
        return

    if args.output is None:
        args.output = str(path)

    images_to_video2(imgs, args.output)


if __name__ == "__main__":
    main()
