import cv2
import mirror
from pathlib import Path
from cv2.typing import MatLike
from typing import Tuple, Literal
import mat
import lines

SAMPLE = './media/Flowers.jpg'

def _get_ex_img(path: str = SAMPLE, desired_h: int = 500) -> MatLike:
    ex_p = Path(path)
    
    if not ex_p.exists():
        raise ValueError(f"{ex_p} not found")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    h, w, _ = img.shape
    
    f = h / desired_h
    
    
    nh, nw = round(h / f), round(w / f)
    
    rimg =  cv2.resize(img, (nh, nw), interpolation=cv2.INTER_CUBIC)
    
    #see original
    cv2.imshow("Original", rimg)
    cv2.waitKey(0)

    return rimg

def test_crop(desired_h:int=500):
    img = _get_ex_img(desired_h=desired_h)
    
    #crop 
    sqr = mirror.crop_square(img)
    cv2.imshow("Square", sqr)
    cv2.waitKey(0)

    return sqr
       

def _project_diag_1chan(
    img: MatLike,
    start: Tuple[int, int],
    end: Tuple[int, int],
    loc: Literal["top", "bottom"],
    diag: Literal["+", "-"]
):
    h, w = img.shape
    if diag == "+":
        mr = mat.mir_p(img)
    else:
        mr = mat.mir_n(img)
    lpnts = lines.line_points(start, end)
    for x_row in range(h):
        for x_col in range(w):
            y_row = lpnts[x_col]
            # reflect the loc
            if (loc == "top" and x_row > y_row) or (loc == "bottom" and x_row < y_row):
                img[x_row, x_col] = mr[x_row, x_col]
    return img

def test_pd1c():
    print('here')
    img = test_crop()
    
    b, g, r = cv2.split(img)
    
    h, w = b.shape
    
    bl = (h, 0)
    tr = (0, w)
    
    tl = (0, 0)
    br = (h, w)
    
    _project_diag_1chan(b, bl, tr, loc='top', diag="+")
    _project_diag_1chan(g, bl, tr, loc='top', diag="+")
    _project_diag_1chan(r, bl, tr, loc='top', diag="+")
    nimg = cv2.merge((b,g,r))
    
    cv2.imshow("Blackout1chan for side: top-postive", nimg)
    cv2.waitKey(0)
    
    b, g, r = cv2.split(img)
    
    _project_diag_1chan(b, bl, tr, loc='bottom', diag="+")
    _project_diag_1chan(g, bl, tr, loc='bottom', diag="+")
    _project_diag_1chan(r, bl, tr, loc='bottom', diag="+")
    nimg = cv2.merge((b,g,r))
    
    cv2.imshow("Blackout1chan for side: bottom-postive", nimg)
    cv2.waitKey(0)
    
    b, g, r = cv2.split(img)
    
    _project_diag_1chan(b, tl, br, loc='top', diag="-")
    _project_diag_1chan(g, tl, br, loc='top', diag="-")
    _project_diag_1chan(r, tl, br, loc='top', diag="-")
    nimg = cv2.merge((b,g,r))
    
    cv2.imshow("Blackout1chan for side: top-negative", nimg)
    cv2.waitKey(0)
    
    b, g, r = cv2.split(img)
    
    _project_diag_1chan(b, tl, br, loc='bottom', diag="-")
    _project_diag_1chan(g, tl, br, loc='bottom', diag="-")
    _project_diag_1chan(r, tl, br, loc='bottom', diag="-")
    nimg = cv2.merge((b,g,r))
    
    cv2.imshow("Blackout1chan for side: bottom-negative", nimg)
    cv2.waitKey(0)


def test_p(side=0):
    img = test_crop()

    b, g, r = cv2.split(img)
    h, w = b.shape
    diagonals = lines.make_lines(h, w)

    # nb = mirror._project_1chan(b, side, diagonals)
    cp = img.copy()
    nimg = mirror._project(cp, side, diagonals)

    side_codes = {
        0 : "top",
        1 : "bottom",
        2 : "left", 
        3 : "right",
        4 : "top +slope",
        5 : "bottom +slope",
        6 : "top -slope",
        7 : "bottom -slope" 
    }

    cv2.imshow(f"Project1chan for side: {side_codes[side]}", nimg)
    cv2.waitKey(0)


def spin_mirror(img):
    cv2.imshow("Image", img)
    cv2.waitKey(50)
    cp = img.copy()
    mrs = ["th", "tn", "rv", "bp", "bh", "bn"]
    for i in range(10):
        for line in mrs:
            hm = mirror.half_mirror(cp, line)
            cv2.imshow("Half Mirror", hm)
            key = cv2.waitKey(100)
            if key == 27:
                break
    cv2.destroyAllWindows()

def spin_mirror2(img):
    cv2.destroyAllWindows()
    cp = img.copy()
    mrs = ['th', 'bh', 'lv', 'rv', 'bp', 'tp', 'bn','tn']
   
    for i in range(20):
        for mr in mrs:
            hm = mirror.half_mirror(cp, mr)
            cv2.imshow("Half Mirror", hm)
            key = cv2.waitKey(100)
            if key == 27:
                break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_spin_mirror(desired_h):
    img = test_crop(desired_h)
    
    spin_mirror(img)

def test_spin_mirror2(desired_h):
    img = test_crop(desired_h)
    
    spin_mirror2(img)



def test_half_mirror(desired_h:int=500):
    # img = test_crop(desired_h)
    # img = _get_ex_img('./media/Flowers.jpg', desired_h)
    img = _get_ex_img('./media/Drawing.png', desired_h)
    side_codes = ['th', 'bh', 'lv', 'rv', 'bp', 'tp', 'bn','tn']
    # for c in side_codes:
    #     hm = mirror.half_mirror(img, c, inplace=True)
    #     cv2.imshow("Half mirrored", hm)
    #     cv2.waitKey(0)
    hm = mirror.half_mirror(img, side_codes[6])
    cv2.imshow("half mirror", hm) 
    cv2.waitKey(0)
    cv2.imshow("original", img)
    cv2.waitKey(0)

def multi_mirror_old(img, mrs=["th", "rv", "tp", "tn"], disp=False) -> MatLike:
    hm = img.copy()
    for line in mrs:
        hm = mirror.half_mirror(hm, line)
        if disp:
            cv2.imshow("Half Mirror", hm)
            cv2.waitKey(0)
    return hm

def test_multiMirror(desired_h):
    
    img = test_crop(desired_h)
    # all_mrs = ['th', 'bh', 'lv', 'rv', 'bp', 'tp', 'bn','tn']
    vline = ['lv', 'rv']
    hline = ['th', 'bh']
    pline = ['tp', 'bp']
    nline = ['tn', 'bn']
    combs = []
    i = 0
    for v in vline:
        for h in hline:
            for p in pline:
                for n in nline:
                    print(f"{i} Combination: {[v, h, p, n]}")
                    combs.append([v, h, p, n])  
                    i += 1     
    
    output = Path('./combs/')
    output.mkdir(exist_ok=True)    
    
    for i, comb in enumerate(combs):
        mm = multi_mirror_old(img, disp=False, mrs=comb)
    
        cv2.imshow(f"Combination: {i}", mm)
        cv2.waitKey(100)
        
        outp = output / f"{i}.png"
        cv2.imwrite(str(outp), mm)

    '''
    Interestingly, there are only 8 unique combinations. They seem to exist in permutation pairs
    
    
                                                    True effect:
    0 Combination: ['lv', 'th', 'tp', 'tn']         0
    1 Combination: ['lv', 'th', 'tp', 'bn']         1
    2 Combination: ['lv', 'th', 'bp', 'tn']         1
    3 Combination: ['lv', 'th', 'bp', 'bn']         0
    4 Combination: ['lv', 'bh', 'tp', 'tn']         4
    5 Combination: ['lv', 'bh', 'tp', 'bn']         5 
    6 Combination: ['lv', 'bh', 'bp', 'tn']         5
    7 Combination: ['lv', 'bh', 'bp', 'bn']         4
    8 Combination: ['rv', 'th', 'tp', 'tn']         8
    9 Combination: ['rv', 'th', 'tp', 'bn']         9
    10 Combination: ['rv', 'th', 'bp', 'tn']        9
    11 Combination: ['rv', 'th', 'bp', 'bn']        8
    12 Combination: ['rv', 'bh', 'tp', 'tn']        12
    13 Combination: ['rv', 'bh', 'tp', 'bn']        13
    14 Combination: ['rv', 'bh', 'bp', 'tn']        13
    15 Combination: ['rv', 'bh', 'bp', 'bn']        12
    '''

def test_multi_mirror2(desired_h):
    img = test_crop(desired_h)
    # all_mrs = ['th', 'bh', 'lv', 'rv', 'bp', 'tp', 'bn','tn']
    mrs = ['lv', 'th', 'tp', 'tn']
    combs = []
    i = 0
    for pos0 in mrs:
        for pos1 in mrs:
            for pos2 in mrs:
                for pos3 in mrs:
                    if len({pos0, pos1, pos2, pos3}) == 4:  
                        combs.append([pos0, pos1, pos2, pos3])
                        print(f"{i} Combination: {[pos0, pos1, pos2, pos3]}")
                        i += 1
                        
    output = Path('./combs2/')
    output.mkdir(exist_ok=True)    
    
    for i, comb in enumerate(combs):
        mm = multi_mirror_old(img, disp=False, mrs=comb)
    
        cv2.imshow(f"Combination: {i}", mm)
        # cv2.imshow("Miror", mm)
        cv2.waitKey(0)
        
        outp = output / f"{i}.png"
        cv2.imwrite(str(outp), mm)

    ''' No new images aside from the original combinations in test1. 
    Additionally, it seems h and v, and p and n, lines need to be paired together (in either order) to produce symmetry
    0 Combination: ['lv', 'th', 'tp', 'tn']
    1 Combination: ['lv', 'th', 'tn', 'tp']
    2 Combination: ['lv', 'tp', 'th', 'tn']
    3 Combination: ['lv', 'tp', 'tn', 'th']
    4 Combination: ['lv', 'tn', 'th', 'tp']
    5 Combination: ['lv', 'tn', 'tp', 'th']
    6 Combination: ['th', 'lv', 'tp', 'tn']
    7 Combination: ['th', 'lv', 'tn', 'tp']
    8 Combination: ['th', 'tp', 'lv', 'tn']
    9 Combination: ['th', 'tp', 'tn', 'lv']
    10 Combination: ['th', 'tn', 'lv', 'tp']
    11 Combination: ['th', 'tn', 'tp', 'lv']
    12 Combination: ['tp', 'lv', 'th', 'tn']
    13 Combination: ['tp', 'lv', 'tn', 'th']
    14 Combination: ['tp', 'th', 'lv', 'tn']
    15 Combination: ['tp', 'th', 'tn', 'lv']
    16 Combination: ['tp', 'tn', 'lv', 'th']
    17 Combination: ['tp', 'tn', 'th', 'lv']
    18 Combination: ['tn', 'lv', 'th', 'tp']
    19 Combination: ['tn', 'lv', 'tp', 'th']
    20 Combination: ['tn', 'th', 'lv', 'tp']
    21 Combination: ['tn', 'th', 'tp', 'lv']
    22 Combination: ['tn', 'tp', 'lv', 'th']
    23 Combination: ['tn', 'tp', 'th', 'lv']
    '''

def test_mirs():
    from mat import mir_p_1chan, mir_n_1chan, mir_p, mir_n
    img = cv2.imread('media/Drawing.png', cv2.IMREAD_COLOR)
    
    cv2.imshow("original", img)
    cv2.waitKey(0)
    
    mirp = mir_p(img)
    
    cv2.imshow("mir_p", mirp)
    cv2.waitKey(0)

    b, g, r = cv2.split(img)
    b, g, r = mir_p_1chan(b), mir_p_1chan(g), mir_p_1chan(r)
    
    mirp2 = cv2.merge((b, g, r))
    cv2.imshow("mir_p1chan", mirp2)
    cv2.waitKey(0)

    mirn = mir_n(img)
    
    cv2.imshow("mir_n", mirn)
    cv2.waitKey(0)
    
    b, g, r = cv2.split(img)
    b, g, r = mir_n_1chan(b), mir_n_1chan(g), mir_n_1chan(r)
    
    mirn2 = cv2.merge((b, g, r))
    cv2.imshow("mir_n1chan", mirn2)
    cv2.waitKey(0)


if __name__ == '__main__':
    # test_make_horiz()
    # test_remove_horiz()
    # test_half_mirror()
    # test_bo1cd()
    # test_pd1c()
    # test_p1c(int(sys.argv[1]))
    # test_remove_diag_p()
    test_half_mirror()
    # test_spin_mirror2(1500)
    # test_multi_mirror2(1500)
    # test_mirs()
    
    cv2.destroyAllWindows()
        