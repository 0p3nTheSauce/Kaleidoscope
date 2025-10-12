#!/usr/bin/env python3

import cv2
import os
import subprocess
from pathlib import Path
from cv2.typing import MatLike
from typing import List, Optional
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

def images_to_video(
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


def convert_mp4(vid_path: str, out_path:Optional[str] = None) -> None:
    """Re-encode .mp4 files created by images_to_video for what's app compatibility

    Args:
        vid_path (str): Path to video file
        out_path (Optional[str], optional): Output path, othewise replaces video if None. Defaults to None.
    """
    # re-encode the .mp4

    original_file = vid_path

    if out_path:
        temp_file = out_path
    else:
        temp_file = vid_path

    if temp_file == original_file:
        temp_file = f"{temp_file.rsplit('.')[0]}_temp.mp4"

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
        if out_path:
            print(f"Video successfully written at {out_path}")
        else:
            os.replace(temp_file, original_file)
            print(f"Video successfully written at {original_file}")
    except subprocess.CalledProcessError as e:
        print("An error occurred while processing the video:", e)
        # Clean up temp file if there was an error
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    parser = ArgumentParser(description="Create and modify videos")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    create_parser = subparsers.add_parser(
        "create", help="Create a .mp4 video from a folder of images"
    )
    create_parser.add_argument(
        "img_dir", type=str, help="Path to folder containing images"
    )
    create_parser.add_argument(
        "-sx", "--suffix", type=str, help="Image suffix.", default=".jpg"
    )
    create_parser.add_argument(
        "-ot", "--output", type=str, help="Output path", default=None
    )
    create_parser.add_argument(
        "-rc",
        "--recode",
        action="store_true",
        help="Recode the .mp4 with FFMPEG (WhatsApp compatibility)",
    )

    recode_parser = subparsers.add_parser(
        "recode", help="Recode the .mp4 with FFMPEG (WhatsApp compatibility)"
    )
    recode_parser.add_argument("vid_path", type=str, help="Path to .mp4 file to recode")
    recode_parser.add_argument(
        "-ot", "--output", type=str, help="Output path", default=None
    )
    
    
    args = parser.parse_args()

    if args.command == 'create':
        
        img_dir = Path(args.img_dir)
        
        if not img_dir.exists():
            print(f"{img_dir} not found")
            return  
        if not img_dir.is_dir():
            print(f"{img_dir} is not a directory")
            return

        imgs = read_dir(img_dir, suff=args.suffix)

        if len(imgs) == 0:
            print(f"no images found for directory: {img_dir} with image suffix: {args.suffix}")
            return

        if args.output is None:
            args.output = str(img_dir)

        images_to_video(imgs, args.output)
        
        if args.recode:
            convert_mp4(args.output)

    elif args.command == 'recode':
        
        vid_path = Path(args.vid_path)
        
        if not vid_path.exists():
            print(f"{vid_path} not found")
            return
        if not vid_path.is_file():
            print(f"{vid_path} is not a file")
            return
        
        convert_mp4(str(vid_path), args.output)
        
        
        

if __name__ == "__main__":
    main()
