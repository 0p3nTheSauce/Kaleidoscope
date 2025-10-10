#!/usr/bin/env python3
import cv2
from cv2.typing import MatLike
from typing import Tuple, List
import numpy as np
import sys
from numba import njit
import os
import argparse
from pathlib import Path
# local
from mat import mir_p, mir_n
from rot import spin_func
from videos import makeVideo


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
        return cv2.flip(img, 0)
    elif line == 1:  # horizontal
        return cv2.flip(img, 1)
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
def blackout_1chan(img: MatLike, side: int) -> MatLike:
    """Convert all pixels to black on the specified side of the image. 
    Works on single channel (grayscale) images.

    Args:
        img (MatLike): Image (h, w) (square).
        side (int): 0 - top vertical,
                    1 - bottom vertical,
                    2 - right horizontal,
                    3 - left horizontal,
                    4 - bottom positive slope,
                    5 - top positive slope,
                    6 - bottom negative slope,
                    7 - top negative slope.

    Raises:
        ValueError: If side provided not one of [0,7]

    Returns:
        MatLike: Image (h, w) with pixels on one side converted to black.
    """
    h, w = img.shape
    bl = img.copy()
    b, t = 1, 0
    m, c, o = 0, 0, 0
    match side:
        case 0:  # top vertical
            bl[h // 2 :, :] = 0
        case 1:  # bottom vertical
            bl[: h // 2, :] = 0
        case 2:  # left horizontal
            bl[:, w // 2 :] = 0
        case 3:  # right horizontal
            bl[:, : w // 2] = 0
        case 4:  # bottom positive slope
            m, c, o = 1, 0, b
        case 5:  # top positive slope
            m, c, o = 1, 0, t
        case 6:  # bottom negative slope
            m, c, o = -1, h, t
        case 7:  # top negative slope
            m, c, o = -1, h, b
        case _:
            raise ValueError("Invalid line code")

    # handle diagonal lines
    if side >= 4:
        # the next two lines are because the image vertical axis is
        # flipped relative to the cartesian plain.
        c = h - c
        m = -m
        for y in range(h):
            for x in range(w):
                yl = m * x + c
                if (o == b and y <= yl) or (o == t and y >= yl):
                    bl[x, y] = 0

    return bl


def blackout(img: MatLike, side: int) -> MatLike:
    """Convert all pixels to black on the specified side of the image.

    Args:
        img (MatLike): Image (h, w, c) (square).
        side (int): 0 - top vertical,
                    1 - bottom vertical,
                    2 - right horizontal,
                    3 - left horizontal,
                    4 - bottom positive slope,
                    5 - top positive slope,
                    6 - bottom negative slope,
                    7 - top negative slope.

    Raises:
        ValueError: If side provided not one of [0,7]

    Returns:
        MatLike: Image (h, w, ck) with pixels on one side converted to black.
    """
    if side not in range(8):
        raise ValueError(f"Side {side} not in range [0,7]")
    
    b, g, r = cv2.split(img)
    b = blackout_1chan(b, side)
    g = blackout_1chan(g, side)
    r = blackout_1chan(r, side)
    return cv2.merge((b, g, r))


def make_diag_n(img: MatLike, colour: Tuple[int, int, int]=(0, 255, 0), disp: bool=False) -> MatLike:
    """Make diagnol line from the top left corner to the bottom right corner.

    Args:
        img (MatLike): Image (h, w, c) (square)
        colour (Tuple[int, int, int], optional): Colour of diagnol. Defaults to (0, 255, 0) (green).
        disp (bool, optional): Display the image with cv2.imshow. Defaults to False.

    Returns:
        MatLike: The image with a diagonal line drawn down the negative diagonal
    """
    h, w, _ = img.shape
    cp = img.copy()
    for rows in range(h):
        for columns in range(w):
            if rows == columns:
                cp[rows, columns] = colour
    if disp:
        cv2.imshow("diag", cp)
        cv2.waitKey(0)
    return cp


def make_diag_p(img: MatLike, colour: Tuple[int, int, int]=(0, 255, 0), disp: bool=False) -> MatLike:
    """Make diagnol line from the bottom left corner to the top right corner.

    Args:
        img (MatLike): Image (h, w, c) (square)
        colour (Tuple[int, int, int], optional): Colour of diagnol. Defaults to (0, 255, 0) (green).
        disp (bool, optional): Display the image with cv2.imshow. Defaults to False.

    Returns:
        MatLike: The image with a diagnol line drawn down the positive diagonal
    """
    h, w, _ = img.shape
    cp = img.copy()
    for rows in range(h):
        for columns in range(w):
            if (h - 1 - rows) == columns:
                cp[rows, columns] = colour
    if disp:
        cv2.imshow("diag", cp)
        cv2.waitKey(0)
    return cp


@njit(cache=True)
def med_of(k: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Calculate the median pixel value out of a list of pixel values.

    Args:
        k (List[Tuple[int, int, int]]): List of pixel values

    Returns:
        Tuple[int, int, int]: Median pixel values
    """
    
    # Separate channels manually for each color dimension
    r = np.array([px[0] for px in k])
    g = np.array([px[1] for px in k])
    b = np.array([px[2] for px in k])

    # Find the median for each channel
    med_r = np.median(r)
    med_g = np.median(g)
    med_b = np.median(b)

    return (round(med_r), round(med_g), round(med_b))

@njit(cache=True)
def neighbours(img: MatLike, coord: Tuple[int,int]) -> List[Tuple[int, int, int]]:
    """Get the pixel values within a 3x3 square centred at coord.

    Args:
        img (MatLike): Image (h, w, c).
        coord (Tuple[int,int]): Coordinate (row, column)

    Returns:
        List[Tuple[int, int, int]]: Neighbouring set of pixel values (b, g, r)
    """
    row, col = coord
    return [
        img[row - 1, col - 1],
        img[row - 1, col],
        img[row - 1, col + 1],
        img[row, col - 1],
        img[row, col],
        img[row, col + 1],
        img[row + 1, col - 1],
        img[row + 1, col],
        img[row + 1, col + 1],
    ]

@njit(cache=True)
def remove_diag_n(img):
    # removes diagnol lines (negative gradient) by taking the median
    # of the neighbouring pixels
    h, w, _ = img.shape
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if row == col:
                k = neighbours(img, (row, col))
                med = med_of(k)
                img[row, col] = med
    return img

@njit(cache=True)
def remove_diag_p(img):
    # removes diagnol lines (positive gradient) by taking the median
    # of the neighbouring pixels
    h, w, _ = img.shape
    for rows in range(1, h - 1):
        for cols in range(1, w - 1):
            if (h - rows - 1) == cols:
                k = neighbours(img, (rows, cols))
                med = med_of(k)
                img[rows, cols] = med
    return img

def half_mirror(img, side, disp=False):
    # mirrors half the image onto the other half
    side_codes = {
        "tv": 0,
        "bv": 1,
        "lh": 2,
        "rh": 3,
        "bp": 4,
        "tp": 5,
        "bn": 6,
        "tn": 7,
    }
    line_codes = {"v": 0, "h": 1, "p": 2, "n": 3}
    sqr = crop_square(img)
    bl = blackout(sqr, side_codes[side])
    ln = side[1]
    mr = mirror(bl, line_codes[ln])

    if disp:
        cv2.imshow("Square", sqr)
        cv2.waitKey(0)
        cv2.imshow("Blackout", bl)
        cv2.waitKey(0)
        cv2.imshow("Mirror", mr)
        cv2.waitKey(0)
    w = cv2.addWeighted(bl, 1.0, mr, 1.0, 0)
    if ln == "p":
        w = remove_diag_p(w)
    elif ln == "n":
        w = remove_diag_n(w)
    return w


def spin_mirror(img):
    cv2.imshow("Image", img)
    cv2.waitKey(50)
    cp = img.copy()
    mrs = ["lh", "tp", "tv", "tn", "rh", "bp", "bv", "bn"]
    for i in range(10):
        for line in mrs:
            hm = half_mirror(cp, line)
            cv2.imshow("Half Mirror", hm)
            key = cv2.waitKey(100)
            if key == 27:
                break
    cv2.destroyAllWindows()


def multi_mirror(img, mrs=["tn", "tp", "tv", "rh"], disp=False):
    hm = img.copy()
    for line in mrs:
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
    img = cv2.imread(f"{in_img}.png")
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
    main()
