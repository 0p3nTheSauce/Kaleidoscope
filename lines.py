"""Implement Bresenhams line algorithm"""

import cv2
import numpy as np
from typing import Tuple
from numba import njit

@njit(cache=True)
def line_points(start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """Generate a dictionary of coordinates on a line. Uses the Bresenhams line algorithm. 
    Uses cv style coordinate indexing (bottom right corner = (h, w))

    Args:
        start (Tuple[int, int]): Starting coordinate
        end (Tuple[int, int]): Final coordinate

    Returns:
        np.ndarray: Index with column index to get row index.
    """
    row1, col1 = start
    row2, col2 = end
    
    # Determine the size needed
    max_col = max(col1, col2)
    edge = np.zeros(max_col + 1, dtype=np.int64)
    
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

def make_lines(h: int,w: int ) -> np.ndarray:
    """Premake the positive and negative diagonal lines for half_mirror

    Args:
        h (int): Image height
        w (int): Image width
        

    Returns:
        Dict[str, List[int]]: '+' for positive diagonal, '-' for negative. The line indexed with column returns row
    """
    tl = (0,0)
    tr = (0, w)
    bl = (h, 0)
    br = (h, w)
    
    pos_diag = line_points(bl, tr)
    neg_diag = line_points(tl, br)
    return np.vstack((pos_diag, neg_diag))
    
    

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
    

if __name__ == "__main__":
    main()
