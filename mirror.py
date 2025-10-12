import cv2
from cv2.typing import MatLike
from typing import List, Literal
import numpy as np
import sys
from numba import njit
import os
# import argparse
# from pathlib import Path

# local
from mat import mir_p, mir_n
from rot import spin_func
from videos import makeVideo
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
        line (int): 0 - vertical,
                    1 - horizontal,
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


def half_mirror(img: MatLike, side: str, inplace:bool = False) -> MatLike:
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



def multi_mirror(img, mrs=["th", "rv", "tp", "tn"], disp=False) -> MatLike:
    hm = img.copy()
    for line in mrs:
        hm = half_mirror(hm, line)
        if disp:
            cv2.imshow("Half Mirror", hm)
            cv2.waitKey(0)
    return hm

def multi_mirror_combs(img, disp=False):
    combs = [
        ['lv', 'th', 'tp', 'tn'],
        ['lv', 'th', 'tp', 'bn'],
        ['lv', 'bh', 'tp', 'tn'],
        ['lv', 'bh', 'tp', 'bn'],
        ['rv', 'th', 'tp', 'tn'],
        ['rv', 'th', 'tp', 'bn'],
        ['rv', 'bh', 'tp', 'tn'],
        ['rv', 'bh', 'tp', 'bn'],
    ]
    multimgs = []
    for c in combs:
        multimgs.append(multi_mirror(img,c,disp))
    return multimgs

def all_dir(input_dir, output_dir, size=(1920, 1080), disp=False):
    idx = 0
    for f in os.listdir(input_dir):
        img_path = os.path.join(input_dir, f)
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        img = crop_square(img)
        spin_func(img, multi_mirror, time=1, outfolder=output_dir, index=idx)
        idx += 360
    cv2.destroyAllWindows()


def main_old():
    # in_img = ''
    in_img = "src_imgs/mario"
    out = ""
    out_w = 0
    out_h = 0
    if len(sys.argv) >= 3:
        in_img = sys.argv[1]
        out = sys.argv[2]
    else:
        print("Usage: <in_img> <out_dir> [out_width] [out_height]")
        sys.exit(1)
    if len(sys.argv) == 5:
        out_w = int(sys.argv[3])
        out_h = int(sys.argv[4])
    img = cv2.imread(in_img)
    if out_w > 0 and out_h > 0:
        img = cv2.resize(img, (out_w, out_h))
    print(f"{in_img}.png")
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    # spin_func(track, edgey_sing, iter=1000, deg=1, time=20)
    spin_func(img, multi_mirror, time=1, outfolder=out)  # very cool
    # spin_func(img, multi_mirror, time=1)
    makeVideo(out)
    # edgey(img ,time=20)

    # all_dir('src_imgs', 'all_src')
    # makeVideo('all_src')
    # hm = multi_mirror(img, disp=True)

    cv2.destroyAllWindows()
    # spin_mirror(img)


def main():
    pass


if __name__ == "__main__":
    main_old()
