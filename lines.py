"""Implement Bresenhams line algorithm"""

import cv2
import numpy as np
import math
from typing import Tuple, Dict
from numba import njit

@njit(cache=True)
def line_points(start: Tuple[int, int], end: Tuple[int, int]) -> Dict[int, int]:
    """Generate a dictionary of coordinates on a line. Uses the Bresenhams line algorithm. Uses cv style coordinate indexing (bottom right corner = (h, w))

    Args:
        start (Tuple[int, int]): Starting coordinate
        end (Tuple[int, int]): Final coordinate

    Returns:
        Dict[int, int]: Hashmap of col : row pairs.
    """
    edge = {}
    row1, col1 = start
    row2, col2 = end
    dx = abs(row2 - row1)
    dy = abs(col2 - col1)
    if row1 < row2:
        sx = 1
    else:
        sx = -1
    if col1 < col2:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while True:
        edge[col1] = row1
        if row1 == row2 and col1 == col2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            row1 = row1 + sx
        if e2 < dx:
            err = err + dx
            col1 = col1 + sy
    return edge


def bresenham_line(start, end, screen, colour=(0, 0, 0), speed=100, slowness=1):
    """draw a line from start to end on the screen"""
    row1, col1 = start
    row2, col2 = end
    count = 0
    dx = abs(row2 - row1)
    dy = abs(col2 - col1)
    if row1 < row2:
        sx = 1
    else:
        sx = -1
    if col1 < col2:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while True:
        screen[col1, row1] = colour
        if count == speed:
            cv2.imshow("Sparrow Screen", screen)
            key = cv2.waitKey(slowness)
            if key == 27:
                break
            count = 0
        else:
            count += 1
        if row1 == row2 and col1 == col2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            row1 = row1 + sx
        if e2 < dx:
            err = err + dx
            col1 = col1 + sy
    return screen


def edge_dir(edge):
    """assumes edge is a collection of points"""
    x0, y0 = edge[0]
    row1, col1 = edge[-1]
    slope = (col1 - y0) / (row1 - x0)
    intercept = col1 - (slope * row1)
    return slope, intercept


def main():
    blank = np.ones((600, 600, 3)) * 255
    start = (500, 10)
    end = (0, 590)
    blank = bresenham_line(start, end, blank)
    cv2.imshow("bresenham", blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lpts = line_points(start, end)

if __name__ == "__main__":
    main()
