import cv2
import mirror
from pathlib import Path
from cv2.typing import MatLike
from typing import Tuple
import numpy as np

SAMPLE = './media/Flowers.jpg'

def _get_ex_img(path: str = SAMPLE) -> MatLike:
    ex_p = Path(path)
    
    if not ex_p.exists():
        raise ValueError(f"{ex_p} not found")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
     
    f = 6.528
    
    h, w, _ = img.shape
    nh, nw = round(h / f), round(w / f)
    
    rimg =  cv2.resize(img, (nh, nw), interpolation=cv2.INTER_CUBIC)
    
    #see original
    cv2.imshow("Original", rimg)
    cv2.waitKey(0)

    return rimg

def test_crop():
    img = _get_ex_img()
    
    #crop 
    sqr = mirror.crop_square(img)
    cv2.imshow("Square", sqr)
    cv2.waitKey(0)

    return sqr

def test_blackout1chan(inplace=False):
    img = test_crop()
    
    b, g, r = cv2.split(img)
    
    for side in [4]:
    
        nb = mirror.blackout_1chan(b, side, inplace)
        ng = mirror.blackout_1chan(g, side, inplace)
        nr = mirror.blackout_1chan(r, side, inplace)
        
        nimg = cv2.merge((nb,ng,nr))
        
        cv2.imshow(f"Blackout1chan for side: {side}", nimg)
        cv2.waitKey(0)
        
def test_bo1cd():
    img = test_crop()
    
    b, g, r = cv2.split(img)
    
    h, w = b.shape
    
    bl = (h, 0)
    tr = (0, w)
    
    nb = mirror._blackout_1chan_diag(b, bl, tr, loc='top')
    ng = mirror._blackout_1chan_diag(g, bl, tr, loc='top')
    nr = mirror._blackout_1chan_diag(r, bl, tr, loc='top')
    
    nimg = cv2.merge((nb,ng,nr))
        
    cv2.imshow("Blackout1chan for side: top-positive", nimg)
    cv2.waitKey(0)
    
def test_p1cd():
    img = test_crop()
    
    b, g, r = cv2.split(img)
    
    h, w = b.shape
    
    bl = (h, 0)
    tr = (0, w)
    
    nb = mirror._project_1chan_diag(b, bl, tr, loc='top')
    ng = mirror._project_1chan_diag(g, bl, tr, loc='top')
    nr = mirror._project_1chan_diag(r, bl, tr, loc='top')
    
    nimg = cv2.merge((nb,ng,nr))
        
    cv2.imshow("Blackout1chan for side: top-positive", nimg)
    cv2.waitKey(0)
    
def test_p1c():
    img = test_crop()
    side = 3
    b, g, r = cv2.split(img)

    nb = mirror._project_1chan(b, side)
    ng = mirror._project_1chan(g, side)
    nr = mirror._project_1chan(r, side)

    nimg = cv2.merge((nb, ng, nr))

    cv2.imshow("Project1chan for side: top-vertical", nimg)
    cv2.waitKey(0)

def test_blackout():
    img = test_crop()
    
    for side in range(8):
        
        nimg = mirror.blackout(img, side)
        
        cv2.imshow(f"Blackout for side: {side}", nimg)
        cv2.waitKey(0)

def make_diag_p(
    img: MatLike,
    colour: Tuple[int, int, int] = (0, 255, 0),
    disp: bool = False,
    inplace: bool = True,
) -> MatLike:
    """Make diagnol line from the bottom left corner to the top right corner.

    Args:
        img (MatLike): Image (h, w, c) (square)
        colour (Tuple[int, int, int], optional): Colour of diagnol. Defaults to (0, 255, 0) (green).
        disp (bool, optional): Display the image with cv2.imshow. Defaults to False.
        inplace (bool, optional): Modify image inplace. Defaults to True.
        
    Returns:
        MatLike: The image with a diagnol line drawn down the positive diagonal
    """
    h, w, _ = img.shape
    if inplace:
        diaged = img
    else:
        diaged = img.copy()

    for rows in range(h):
        for columns in range(w):
            if (h - 1 - rows) == columns:
                diaged[rows, columns] = colour
    if disp:
        cv2.imshow("diag", diaged)
        cv2.waitKey(0)
    return diaged

def make_diag_n(
    img: MatLike,
    colour: Tuple[int, int, int] = (0, 255, 0),
    disp: bool = False,
    inplace: bool = True,
) -> MatLike:
    """Make diagnol line from the top left corner to the bottom right corner.

    Args:
        img (MatLike): Image (h, w, c) (square)
        colour (Tuple[int, int, int], optional): Colour of diagnol. Defaults to (0, 255, 0) (green).
        disp (bool, optional): Display the image with cv2.imshow. Defaults to False.
        inplace (bool, optional): Modify image inplace. Defaults to True.

    Returns:
        MatLike: The image with a diagonal line drawn down the negative diagonal
    """
    h, w, _ = img.shape
    if inplace:
        diaged = img
    else:
        diaged = img.copy()

    for rows in range(h):
        for columns in range(w):
            if rows == columns:
                diaged[rows, columns] = colour
    if disp:
        cv2.imshow("diag", diaged)
        cv2.waitKey(0)
    return diaged

def make_horiz(
    img: MatLike,
    colour: Tuple[int, int, int] = (0,255,0),
    disp: bool=False,
    inplace: bool =True,
) -> MatLike:

    h, w, _ = img.shape
    
    if inplace:
        horz = img
    else:
        horz = img.copy()
        
    centre = h // 2
    
    for col in range(1, w - 1):
        horz[centre, col] = colour
        
    if disp:
        cv2.imshow("Horiz", horz)
        cv2.waitKey(0)

    return horz

def test_make_horiz():
    img = test_crop()
    make_horiz(img, disp=True)
    return img

def test_make_diag_n(inplace=True):
    img = test_crop()
    diag = make_diag_n(img, disp=True, inplace=inplace)
    return diag

def test_make_diag_p():
    img = test_crop()
    return make_diag_p(img, disp=True)

def test_remove_diag_n():
    img = test_make_diag_n()
    remd = mirror.remove_diag_n(img)
    cv2.imshow("Removed diagonal", remd)
    cv2.waitKey(0)
    

def test_remove_diag_p():
    img = test_make_diag_p()
    remd = mirror.remove_diag_p(img)
    cv2.imshow("Removed diagonal", remd)
    cv2.waitKey(0)
    

def test_remove_horiz():
    img = test_make_horiz()
    #need to make sure that the height is uneven
    h, w, _ = img.shape
    print(img.shape)
    if h % 2 == 0:
        img = np.vstack((np.zeros(w), img ))
    print(img.shape)
    remd = mirror.remove_horiz(img)
    cv2.imshow("Horizontal removed", remd)
    cv2.waitKey(0)
    


def test_half_mirror():
    img = test_crop()
    side_codes = ['tv', 'bv', 'lh', 'rh', 'bp', 'tp', 'bn','tn']
    # for c in side_codes:
    #     hm = mirror.half_mirror(img, c)
    #     cv2.imshow("Half mirrored", hm)
    #     cv2.waitKey(0)
    hm = mirror.half_mirror(img, side_codes[4], disp=True)
    cv2.imshow("half mirror", hm) 
    cv2.waitKey(0)
    # cv2.imshow("original", img)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # test_make_horiz()
    # test_remove_horiz()
    # test_half_mirror()
    # test_bo1cd()
    test_p1c()
    # test_remove_diag_p()
    # test_blackout1chan()
    cv2.destroyAllWindows()
        