import cv2
import mirror
from pathlib import Path
from cv2.typing import MatLike

def _get_ex_img(path: str = './media/Flowers.jpg') -> MatLike:
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

def test_blackout1chan():
    img = test_crop()
    
    b, g, r = cv2.split(img)
    
    for side in range(8):
    
        nb = mirror.blackout_1chan(b, side)
        ng = mirror.blackout_1chan(g, side)
        nr = mirror.blackout_1chan(r, side)
        
        nimg = cv2.merge((nb,ng,nr))
        
        cv2.imshow(f"Blackout1chan for side: {side}", nimg)
        cv2.waitKey(0)
        
def test_blackout():
    img = test_crop()
    
    for side in range(8):
        
        nimg = mirror.blackout(img, side)
        
        cv2.imshow(f"Blackout for side: {side}", nimg)
        cv2.waitKey(0)

def test_make_diag_n():
    img = test_crop()
    return mirror.make_diag_n(img, disp=True)

def test_make_diag_p():
    img = test_crop()
    return mirror.make_diag_p(img, disp=True)

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
    
def test_remove_diag_n_t():
    #seems to have no effect 
    img = test_make_diag_n()
    remd = mirror.remove_diag_n_t(img)
    cv2.imshow("Removed diagonal (using neighbours_p)", remd)
    cv2.waitKey(0)

def test_remove_diag_p_t():
    #flipps the added diagonal 
    img = test_make_diag_p()
    remd = mirror.remove_diag_p_t(img)
    cv2.imshow("Removed diagonal (using neighbours_n)", remd)
    cv2.waitKey(0)

if __name__ == '__main__':
    test_remove_diag_n_t()
    cv2.destroyAllWindows()
        