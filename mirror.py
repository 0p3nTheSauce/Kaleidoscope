import cv2
from cv2.typing import MatLike
from typing import List, Literal, Optional
from functools import partial
import numpy as np
from numba import njit
import os
from argparse import ArgumentParser
from pathlib import Path

# local
from mat import mir_p, mir_n
from rot import spin_func
from videos import images_to_video, convert_mp4
import lines

# constants
CV_FLIP_VERT = 0
CV_FLIP_HORIZ = 1


def crop_square(img: MatLike) -> MatLike:
    """Centre crops the largest square from an image

    Args:
        img (MatLike): Image (h, w, c).

    Returns:
        MatLike: The largest centred square crop from the original image.
    """
    h, w, _ = img.shape
    if h > w:
        diff = h - w
        dif1 = diff // 2
        dif2 = diff - dif1
        return img[dif1 : h - dif2, :]
    diff = w - h
    dif1 = diff // 2
    dif2 = diff - dif1
    return img[:, dif1 : w - dif2]


def mirror(img: MatLike, line: int) -> MatLike:
    """Mirror an image along the specified line.

    Args:
        img (MatLike): Image (h, w, c) (square).
        line (int): 0 - horizontal
                    1 - vertical.
                    2 - postive incline diagonal,
                    3 - negative incline diagonal.

    Raises:
        ValueError: If line is not one of [0,3].

    Returns:
        MatLike: Image mirrored along line.
    """
    if line == 0:  # vertical
        return cv2.flip(img, CV_FLIP_VERT)
    elif line == 1:  # horizontal
        return cv2.flip(img, CV_FLIP_HORIZ)
    elif line == 2:  # positive incline diagonal
        b, g, r = cv2.split(img)
        b = mir_p(b)
        g = mir_p(g)
        r = mir_p(r)
        return cv2.merge((b, g, r))
    elif line == 3:  # negative incline diagonal
        b, g, r = cv2.split(img)
        b = mir_n(b)
        g = mir_n(g)
        r = mir_n(r)
        return cv2.merge((b, g, r))
    else:
        raise ValueError(f"Provided: {line} not one of available lines: 0, 1, 2, 3")


@njit(cache=True)
def _project_diag_1chan(
    img: MatLike,
    loc: Literal["top", "bottom"],
    diag: Literal["+", "-"],
    lpnts: List[int],
) -> MatLike:
    """Reflect one half of an image onto the other side (not inplace).

    Args:
        img (MatLike): Image (h, w)
        loc (Literal['top', 'bottom']): Location with respect to dividing line.
        diag (Literal['+', '-']): Sign of the diagonal gradient.

    Returns:
        MatLike: Image with 1 plane of symmetry.
    """
    h, w = img.shape
    if diag == "+":
        mr = mir_p(img)
    else:
        mr = mir_n(img)

    for x_row in range(h):
        for x_col in range(w):
            y_row = lpnts[x_col]
            # reflect the loc
            if (loc == "top" and x_row > y_row) or (loc == "bottom" and x_row < y_row):
                img[x_row, x_col] = mr[x_row, x_col]
    return img


@njit(cache=True)
def _project_1chan(img: MatLike, side: int, diagonals: np.ndarray) -> MatLike:
    """Reflect one half of an image onto the other side (inplace).

    Args:
        img (MatLike): Image (h, w)
        side (int): 0 - top half,
                    1 - bottom half,
                    2 - right half,
                    3 - left half,
                    4 - bottom positive slope,
                    5 - top positive slope,
                    6 - bottom negative slope,
                    7 - top negative slope.

    Returns:
        MatLike: Image with 1 plane of symmetry.
    """
    h, w = img.shape

    match side:
        case 0:  # reflect top
            mid = h // 2
            img[h - mid :, :] = img[:mid, :][::-1, :]
        case 1:  # reflect bottom
            mid = h // 2
            img[:mid, :] = img[h - mid :, :][::-1, :]
        case 2:  # reflect left
            mid = w // 2
            img[:, w - mid :] = img[:, :mid][:, ::-1]
        case 3:  # reflect right
            mid = w // 2
            img[:, :mid] = img[:, w - mid :][:, ::-1]
        case 4:  # reflect top positive slope
            img = _project_diag_1chan(img, "top", "+", diagonals[0])
        case 5:  # reflect bottom positive slope
            img = _project_diag_1chan(img, "bottom", "+", diagonals[0])
        case 6:  # reflect top negative slope
            img = _project_diag_1chan(img, "top", "-", diagonals[1])
        case 7:  # reflect bottom negative slope
            img = _project_diag_1chan(img, "bottom", "-", diagonals[1])
        case _:
            raise ValueError("Unsupported side code")
    return img


def half_mirror(img: MatLike, side: str, inplace: bool = False) -> MatLike:
    """
    Reflect one half of an image onto the other side (not inplace)

    Args:
        img (MatLike): Image (h, w, c)
        side (str): first letter location, second is plane of symmetry
            - th (top, horizontal)
            - bh (bottom, horizontal)
            - lv (left, vertical)
            - rv (right, vertical)
            - tp (top, positive diagnoal)
            - bp (bottom, negative diagonal)
            - tn (top, negative diagonal)
            - bn (bottom, negative diagonal)

    Returns:
        MatLike: Image with one plane of symmetry
    """
    side_codes = {
        "th": 0,
        "bh": 1,
        "lv": 2,
        "rv": 3,
        "bp": 4,
        "tp": 5,
        "bn": 6,
        "tn": 7,
    }

    b, g, r = cv2.split(img)
    h, w = b.shape
    snum = side_codes[side]
    diagonals = lines.make_lines(h, w)
    b = _project_1chan(b, snum, diagonals)
    g = _project_1chan(g, snum, diagonals)
    r = _project_1chan(r, snum, diagonals)
    if inplace:
        img = cv2.merge((b, g, r), dst=img)
        return img
    else:
        hm = cv2.merge((b, g, r))
        return hm


def multi_mirror(img: MatLike, perm: int = 0, disp: bool = False, comb: Optional[List[str]] = None) -> MatLike:
    """Create one of the 8 possible images produced by applying 4 planes of symmetry.

    Args:
        img (MatLike): Image (h, w, c)
        perm (int, optional): Permutation of operations [0,7]. Defaults to 0.
        disp (bool, optional): Display intermediary images from application of single planes of symmetry. Defaults to False.

    Returns:
        MatLike: One of 8 unique possible images with 4 planes of symmetry.
    """
    combs = [
        ["lv", "th", "tp", "tn"],
        ["lv", "th", "tp", "bn"],
        ["lv", "bh", "tp", "tn"],
        ["lv", "bh", "tp", "bn"],
        ["rv", "th", "tp", "tn"],
        ["rv", "th", "tp", "bn"],
        ["rv", "bh", "tp", "tn"],
        ["rv", "bh", "tp", "bn"],
    ]
    if comb is None:
        comb = combs[perm]

    
    hm = img.copy()
    for line in comb:
        hm = half_mirror(hm, line)
        if disp:
            cv2.imshow("Half Mirror", hm)
            cv2.waitKey(0)
    return hm


def all_dir(input_dir, output_dir, size=(1920, 1080), disp=False):
    idx = 0
    for f in os.listdir(input_dir):
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        img = crop_square(img)
        spin_func(img, multi_mirror, wait=1, outfolder=output_dir, index=idx, disp=disp)
        idx += 360
    cv2.destroyAllWindows()


def main():
    # Image in parser
    img_in_parser = ArgumentParser(add_help=False)
    img_in_parser.add_argument("in_img", help="Path to input image")
    img_in_parser.add_argument(
        "-sq",
        "--square",
        action="store_true",
        help="Crop to centre square before other operations",
    )
    img_in_parser.add_argument(
        "-ns",
        "--new_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="New size as width height (e.g., -ns 1920 1080)",
    )
    img_in_parser.add_argument(
        "-fx", "--factor_x", type=float, help="Factor to multiply width by (resize)"
    )
    img_in_parser.add_argument(
        "-fy", "--factor_y", type=float, help="Factor to multiply height by (resize)"
    )
    img_in_parser.add_argument(
        "-nd", "--no_disp", action="store_true", help="Do not view output"
    )
    img_in_parser.add_argument(
        "-do", "--disp_original", action="store_true", help="View original image"
    )
    #image out parser
    img_out_parser = ArgumentParser(add_help=False)
    img_out_parser.add_argument("-ot", "--out_img", type=str, help="Output path of image. Otherwise don't save")

    # Main parser
    parser = ArgumentParser(
        "mirror.py",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Each subparser inherits from parent
    mirror_parser = subparsers.add_parser(
        "mirror", help="Mirror a whole image", parents=[img_in_parser, img_out_parser]
    )
    mirror_parser.add_argument(
        "plane",
        help="Mirror about this plane of symmetry: "
            "Planes: h/v/p/n (horizontal/vertical/+diagonal/-diagonal)",
        choices=["v", "h", "p", "n"],
    )

    halfm_parser = subparsers.add_parser(
        "half_mirror",
        help="Reflect one half of the image onto the other",
        parents=[img_in_parser, img_out_parser],
    )
    halfm_parser.add_argument(
        "plane",
        choices=["th", "bh", "lv", "rv", "tp", "bp", "tn", "bn"],
        metavar="SIDE+PLANE",
        help="Side to reflect + plane of symmetry (e.g. th). "
        "Sides: t/b/l/r (top/bottom/left/right). "
        "Planes: h/v/p/n (horizontal/vertical/+diagonal/-diagonal). ",
    )

    multim_parser = subparsers.add_parser(
        "multi_mirror",
        help="Create one of the 8 possible images produced by applying 4 planes of symmetry.",
        parents=[img_in_parser, img_out_parser],
    )
    # Provide either perm_num or comb
    perm_group = multim_parser.add_mutually_exclusive_group(required=True)
    perm_group.add_argument(
        "-pn",
        "--perm_num",
        type=int,
        choices=range(8),
        help="Permutation of operations [0-7].",
    )
    
    perm_group.add_argument(
        "-cb",
        "--comb",
        type=str,
        nargs='+',
        metavar="PLANE",
        help="Custom combination of planes (e.g., lv th tp tn)",
    )
    multim_parser.add_argument(
        "-dv", "--disp_verbose", help="Display intermediary steps", action="store_true"
    )

    spin_parser = subparsers.add_parser(
        "spin_mirror",
        help="Rotate image while applying multi-mirror (creates kaleidoscope)",
        parents=[img_in_parser],
    )
    perm_group = spin_parser.add_mutually_exclusive_group(required=True)
    perm_group.add_argument(
        "-pn",
        "--perm_num",
        type=int,
        choices=range(8),
        help="Permutation of operations [0-7].",
    )
    
    perm_group.add_argument(
        "-cb",
        "--comb",
        type=str,
        nargs='+',
        metavar="PLANE",
        help="Custom combination of planes (e.g., lv th tp tn)",
    )
    spin_parser.add_argument(
        '-dv',
        
    )
    spin_parser.add_argument(
        "-it",
        "--iterations",
        type=int,
        help="Number of times to apply function and rotation. Defaults to 360.",
        default=360,
    )
    spin_parser.add_argument(
        "-dg",
        "--degrees",
        type=int,
        help="Degrees to rotate per iteration. Defaults to 1.",
        default=1,
    )
    spin_parser.add_argument(
        "-wt",
        "--wait",
        type=int,
        help="Wait peroid between application (ms). Defaults to 1.",
        default=1,
    )
    spin_parser.add_argument(
        "-ix",
        "--index",
        type=int,
        help="For enumerating image paths. Defaults to 0.",
        default=0,
    )
    spin_parser.add_argument(
        "-od",
        "--out_dir",
        type=str,
        help="Output path. Save intermediary images to a directory",
    )
    spin_parser.add_argument(
        "-ov", "--out_vid", type=str, help="Output path. Create a video from the images"
    )
    spin_parser.add_argument(
        "-fr",
        "--frame_rate",
        type=float,
        help="Frame rate of output video",
        default=30.0,
    )
    spin_parser.add_argument(
        "-vc",
        "--video_code",
        type=str,
        help="Video codec for output video",
        default="mp4v",
    )
    spin_parser.add_argument(
        "-nr",
        "--no_recode",
        action="store_true",
        help="Don't recode the video with FFMPEG",
    )

    args = parser.parse_args()

    in_img = Path(args.in_img)

    if not in_img.exists():
        print(f"{in_img} not found")
        return

    if not in_img.is_file():  # will need to change when adding all_dir
        print(f"{in_img} is not a file")
        return

    img = cv2.imread(str(in_img), cv2.IMREAD_COLOR)

    if args.new_size:
        img = cv2.resize(img, dsize=tuple(args.new_size))
    elif args.factor_x or args.factor_y:
        fx = args.factor_x or 1.0
        fy = args.factor_y or 1.0
        img = cv2.resize(img, dsize=None, fx=fx, fy=fy)

    if args.square:
        img = crop_square(img)

    if args.disp_original:
        cv2.imshow("Original", img)
        cv2.waitKey(0)

    res = img

    if args.command == "mirror":
        dic = {
            "v": 0,
            "h": 1,
            "p": 2,
            "n": 3
        }
        res = mirror(img, dic[args.plane])
    elif args.command == "half_mirror":
        res = half_mirror(img, args.plane)
    elif args.command == "multi_mirror":
        if args.comb:
            res = multi_mirror(img, args.disp_verbose, comb=args.comb)
        else:
            res = multi_mirror(img, args.perm_num, args.disp_verbose)

    if (
        args.command == "mirror"
        or args.command == "half_mirror"
        or args.command == "multi_mirror"
    ):
        if not args.no_disp:
            cv2.imshow(args.command, res)
            cv2.waitKey(0)
            
        if args.out_img:
            cv2.imwrite(args.out_img, res)

    elif args.command == "spin_mirror":
        out_path = None
        if args.out_dir:
            out_path = Path(args.out_dir)
            out_path.mkdir(exist_ok=True, parents=True)
            out_path = str(out_path)

        if args.comb:
            func = partial(multi_mirror, comb=args.comb)
        else:
            func = partial(multi_mirror, perm=args.perm_num)

        res_imgs = spin_func(
            img,
            func,
            iter=args.iterations,
            deg=args.degrees,
            wait=args.wait,
            index=args.index,
            disp=(not args.no_disp),
            outfolder=out_path,
        )

        if args.out_vid:
            out_path = images_to_video(res_imgs, args.out_vid, args.frame_rate, args.video_code)
            if not args.no_recode:
                convert_mp4(out_path)


if __name__ == "__main__":
    main()
