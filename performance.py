#!/usr/bin/env python3
from cv2.typing import MatLike
from numba import njit
from mirror import CV_FLIP_HORIZ
from old.mirror_1 import neighbours, med_of
import timeit
import time
import cv2
import numpy as np

# local imports
from mat import mir_p, mir_n
from old.mat_0 import mir_p_o, mir_n_o
from mirror import mirror
from old.mirror_1 import blackout, remove_diag_n, remove_diag_p

from test import make_diag_n, make_diag_p
from old.mirror_0 import (
    blackout_o,
    blackout_i,
    blackout_m,
    mirror_o,
    remove_diag_n_o,
    remove_diag_p_o,
    blackout_gpt,
)


def test_mirs_img(img, numiter=100):
    # original functions
    o_mir_p = mir_p_o(img)
    o_mir_n = mir_n_o(img)
    print(f"Original mir functions for {numiter} iterations (seconds)")
    cv2.imshow("Original mir_p", o_mir_p)
    cv2.imshow("Original mir_n", o_mir_n)
    cv2.waitKey(0)
    o_mir_p_time = timeit.timeit(lambda: mir_p_o(img), number=numiter)
    o_mir_n_time = timeit.timeit(lambda: mir_n_o(img), number=numiter)
    print(f"Original mir_p time: {o_mir_p_time}")
    print(f"Original mir_n time: {o_mir_n_time}")

    # optimized functions
    mir_img = mir_p(img)
    mir2_img = mir_n(img)
    print()
    print(f"Modified mir functions for {numiter} iterations (seconds)")
    cv2.imshow("mir_p", mir_img)
    cv2.imshow("mir_n", mir2_img)
    cv2.waitKey(0)
    mir_time = timeit.timeit(lambda: mir_p(img), number=numiter)
    mir2_time = timeit.timeit(lambda: mir_n(img), number=numiter)
    print(f"mir_p time: {mir_time}")
    print(f"mir_p time: {mir2_time}")
    """
  Original mir functions for 100 iterations (seconds)
  Original mir_p time: 0.00047601999904145487
  Original mir_n time: 2.157500057364814e-05
  
  Modified mir functions for 100 iterations (seconds)
  mir_p time: 9.301099998992868e-05
  mir_p time: 8.293099745060317e-05"""


def test_sing_blackout(img, func, params, title, numiter=100):
    # single type of blackout function
    tv, bv, lh, rh, bp, tp, bn, tn = params
    bltv = func(img, tv)
    blbv = func(img, bv)
    bllh = func(img, lh)
    blrh = func(img, rh)
    blbp = func(img, bp)
    bltp = func(img, tp)
    blbn = func(img, bn)
    bltn = func(img, tn)

    cv2.imshow(f"{title} tv", bltv)
    cv2.waitKey(0)
    cv2.imshow(f"{title} bv", blbv)
    cv2.waitKey(0)
    cv2.imshow(f"{title} lh", bllh)
    cv2.waitKey(0)
    cv2.imshow(f"{title} rh", blrh)
    cv2.waitKey(0)
    cv2.imshow(f"{title} bp", blbp)
    cv2.waitKey(0)
    cv2.imshow(f"{title} tp", bltp)
    cv2.waitKey(0)
    cv2.imshow(f"{title} bn", blbn)
    cv2.waitKey(0)
    cv2.imshow(f"{title} tn", bltn)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    bltv_time = timeit.timeit(lambda: func(img, tv), number=numiter)
    blbv_time = timeit.timeit(lambda: func(img, bv), number=numiter)
    bllh_time = timeit.timeit(lambda: func(img, lh), number=numiter)
    blrh_time = timeit.timeit(lambda: func(img, rh), number=numiter)
    blbp_time = timeit.timeit(lambda: func(img, bp), number=numiter)
    bltp_time = timeit.timeit(lambda: func(img, tp), number=numiter)
    blbn_time = timeit.timeit(lambda: func(img, bn), number=numiter)
    bltn_time = timeit.timeit(lambda: func(img, tn), number=numiter)

    print()
    print(f"{title} for {numiter} iterations (seconds)")
    print(f"time tv: {bltv_time}")
    print(f"time bv: {blbv_time}")
    print(f"time lh: {bllh_time}")
    print(f"time rh: {blrh_time}")
    print(f"time bp: {blbp_time}")
    print(f"time tp: {bltp_time}")
    print(f"time bn: {blbn_time}")
    print(f"time tn: {bltn_time}")


def test_blackout_img(img, numiter=100):
    # original function
    params_o = ("tv", "bv", "lh", "rh", "bp", "tp", "bn", "tn")
    test_sing_blackout(img, blackout_o, params_o, "Original blackout", numiter)

    # optimized function
    tv, bv, lh, rh, bp, tp, bn, tn = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )  # numba is better with numbers
    params = (tv, bv, lh, rh, bp, tp, bn, tn)
    test_sing_blackout(img, blackout, params, "Modified blackout", numiter)
    """
  Original functions for 100 iterations (seconds)
  Original blackout time tv: 0.39182827599870507
  Original blackout time bv: 0.3818695129994012
  Original blackout time lh: 0.38410538699827157
  Original blackout time rh: 0.3846034720008902
  Original blackout time bp: 33.727699044000474
  Original blackout time tp: 34.71868012699997
  Original blackout time bn: 34.081632714998705
  Original blackout time tn: 33.73179519000041

  Jit compiled functions for 100 iterations (seconds)
  blackout time tv: 0.19463914899824886
  blackout time bv: 0.1747481689999404
  blackout time lh: 0.1698626850011351
  blackout time rh: 0.17183801699866308
  blackout time bp: 0.4557783880009083
  blackout time tp: 0.4518411040007777
  blackout time bn: 0.45001472900185036
  blackout time tn: 0.451004742000805"""


def test_modified_blackouts(img, numiter=100):
    cp, cpi, cpm = img.copy(), img.copy(), img.copy()

    # first partially jit compiled
    tv, bv, lh, rh, bp, tp, bn, tn = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )  # numba is better with numbers
    params = (tv, bv, lh, rh, bp, tp, bn, tn)
    test_sing_blackout(
        cp, blackout, params, "blackoutv2 (partially jit compiled)", numiter
    )

    # fully jit compiled, mutates the image
    test_sing_blackout(
        cpi, blackout_m, params, "blackoutv3_m (fully jit, mutates)", numiter
    )

    # fully jit compiled, immutable
    test_sing_blackout(
        cpm, blackout_i, params, "blackoutv3_i (fully jit, immutable)", numiter
    )

    """
  blackoutv2 (partially jit compiled) for 100 iterations (seconds)
  blackout time tv: 0.2001306870006374
  blackout time bv: 0.18925111899807234
  blackout time lh: 0.19029038100052276
  blackout time rh: 0.19442824100042344
  blackout time bp: 0.4711109599993506
  blackout time tp: 0.5137944570014952
  blackout time bn: 0.45774696100124856
  blackout time tn: 0.46173637000174494

  blackoutv3_m (fully jit, mutates) for 100 iterations (seconds)
  blackout time tv: 0.29457256000023335
  blackout time bv: 0.27850160299931304
  blackout time lh: 0.2808784619992366
  blackout time rh: 0.2802346810021845
  blackout time bp: 0.5555819460023486
  blackout time tp: 0.554230028999882
  blackout time bn: 0.5505143469999894
  blackout time tn: 0.5537112619967957

  blackoutv3_i (fully jit, immutable) for 100 iterations (seconds)
  blackout time tv: 0.32949251500031096
  blackout time bv: 0.31490190399927087
  blackout time lh: 0.32081450400073663
  blackout time rh: 0.32218570400073077
  blackout time bp: 0.5979335299998638
  blackout time tp: 0.5929794860021502
  blackout time bn: 0.5936000490000879
  blackout time tn: 0.6046839369992085"""


def test_sing_mirror(img, func, title, numiter=100, disp=False):
    v, h, p, n = 0, 1, 2, 3

    mirv = func(img, v)
    mirh = func(img, h)
    mirp = func(img, p)
    mirn = func(img, n)

    if disp:
        cv2.imshow(f"{title} v", mirv)
        cv2.waitKey(0)
        cv2.imshow(f"{title} h", mirh)
        cv2.waitKey(0)
        cv2.imshow(f"{title} p", mirp)
        cv2.waitKey(0)
        cv2.imshow(f"{title} n", mirn)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    mirv_time = timeit.timeit(lambda: func(img, v), number=numiter)
    mirh_time = timeit.timeit(lambda: func(img, h), number=numiter)
    mirp_time = timeit.timeit(lambda: func(img, p), number=numiter)
    mirn_time = timeit.timeit(lambda: func(img, n), number=numiter)

    print()
    print(f"{title} for {numiter} iterations (seconds)")
    print(f"time v: {mirv_time}")
    print(f"time h: {mirh_time}")
    print(f"time p: {mirp_time}")
    print(f"time n: {mirn_time}")


def test_mirror(img, numiter=100, disp=False):
    # original functions
    test_sing_mirror(img, mirror_o, "Original mirror", numiter, disp)

    # optimized functions
    test_sing_mirror(img, mirror, "mirror", numiter, disp)

    """
  Original mirror functions for 100 iterations (seconds)
  Original mirror time v: 0.07256410300033167
  Original mirror time h: 0.06661735300440341
  Original mirror time p: 0.488386511002318
  Original mirror time n: 0.4426036499935435

  Modified mirror functions for 100 iterations (seconds)
  mirror time v: 0.043370286002755165
  mirror time h: 0.06154745399544481
  mirror time p: 0.4835698019960546
  mirror time n: 0.4653278540063184"""


def test_remove_diag(img, numiter=100):
    # make images
    diagp = make_diag_p(img)
    diagn = make_diag_n(img)

    cv2.imshow("Diag p", diagp)
    cv2.waitKey(0)
    cv2.imshow("Diag n", diagn)
    cv2.waitKey(0)

    # original functions
    removedp = remove_diag_p_o(diagp)
    removedn = remove_diag_n_o(diagn)

    cv2.imshow("Removed diag p", removedp)
    cv2.waitKey(0)
    cv2.imshow("Removed diag n", removedn)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    removedp_time = timeit.timeit(lambda: remove_diag_p_o(diagp), number=numiter)
    removedn_time = timeit.timeit(lambda: remove_diag_n_o(diagn), number=numiter)

    print()
    print(f"Original remove diag functions for {numiter} iterations (seconds)")
    print(f"Original remove diag time p: {removedp_time}")
    print(f"Original remove diag time n: {removedn_time}")

    # optimized functions
    removedp = remove_diag_p(diagp)
    removedn = remove_diag_n(diagn)

    cv2.imshow("Removed diag p", removedp)
    cv2.waitKey(0)
    cv2.imshow("Removed diag n", removedn)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    removedp_time = timeit.timeit(lambda: remove_diag_p(diagp), number=numiter)
    removedn_time = timeit.timeit(lambda: remove_diag_n(diagn), number=numiter)

    print()
    print(f"Modified remove diag functions for {numiter} iterations (seconds)")
    print(f"Modified remove diag time p: {removedp_time}")
    print(f"Modified remove diag time n: {removedn_time}")

    """
  Original remove diag functions for 100 iterations (seconds)
  Original remove diag time p: 8.11424048399931
  Original remove diag time n: 4.24769181100055

  Modified remove diag functions for 100 iterations (seconds)
  Modified remove diag time p: 0.16073157700157026
  Modified remove diag time n: 0.1617420470029174
  """


def test_gpt_blackout(img, numiter=100):
    tv, bv, lh, rh, bp, tp, bn, tn = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )  # numba is better with numbers
    params = (tv, bv, lh, rh, bp, tp, bn, tn)
    # current best function
    test_sing_blackout(img, blackout, params, "blackoutv2", numiter)

    # chatgpts blackout
    test_sing_blackout(img, blackout_gpt, params, "blackout_gpt", numiter)

    """
  blackoutv2 for 100 iterations (seconds)
  time tv: 0.16602702000091085
  time bv: 0.14297499699750915
  time lh: 0.1346176259976346
  time rh: 0.13557515799766406
  time bp: 0.40778217400293215
  time tp: 0.4083261819978361
  time bn: 0.40803431200038176
  time tn: 0.4033592360028706

  blackout_gpt for 100 iterations (seconds)
  time tv: 0.7605457760000718
  time bv: 0.7576867640018463
  time lh: 0.7704638539980806
  time rh: 0.7995781470017391
  time bp: 1.0594865300008678
  time tp: 1.0276244760025293
  time bn: 1.0329738609980268
  time tn: 1.0311045029993693
  """


def test_remove_diag2(img=None):
    # from mirror import remove_diag_p, remove_diag_p_np, remove_diag_p_o
    def remove_diag_p(img: MatLike) -> MatLike:
        cv2.flip(img, CV_FLIP_HORIZ, dst=img)
        remove_diag_n(img)
        return cv2.flip(img, CV_FLIP_HORIZ, dst=img)

    @njit(cache=True)
    def remove_diag_p_o(img: MatLike) -> MatLike:
        """Removes diagnol lines (positive gradient) by taking the median of the neighbouring pixel values

        Args:
            img (MatLike): Image (h, w, c) (square)
            inplace (bool, optional): Modify image inplace. Defaults to True.

        Returns:
            MatLike: Image with diagonal line removed.
        """
        h, w, _ = img.shape
        for rows in range(1, h - 1):
            for cols in range(1, w - 1):
                if (h - rows - 1) == cols:
                    k = neighbours(img, (rows, cols))
                    med = med_of(k)
                    img[rows, cols] = med
        return img

    @njit(cache=True)
    def remove_diag_p_np(img):
        flipped = img[:, ::-1]  # Just creates a view, no copy
        remove_diag_n(flipped)  # This modifies the view
        return flipped[:, ::-1]  # Returns another view

    # Create test image
    if img is not None:
        test_img = img
    else:
        img_size = (1000, 1000, 3)  # Adjust size as needed
        test_img = np.random.randint(0, 256, img_size, dtype=np.uint8)
    numiter = 1000

    # Warm up numba JIT
    test_copy = test_img.copy()
    remove_diag_p_np(test_copy)
    remove_diag_p_o(test_copy)
    remove_diag_p(test_copy)  # not necessary but why not

    np_time = timeit.timeit(lambda: remove_diag_p_np(test_copy), number=numiter)
    o_time = timeit.timeit(lambda: remove_diag_p_o(test_copy), number=numiter)
    cv_time = timeit.timeit(lambda: remove_diag_p(test_copy), number=numiter)

    print()
    print(f"Modified remove diag functions for {numiter} iterations (seconds)")
    print(f"np_time: {np_time}")
    print(f"o_time: {o_time}")
    print(f"cv_time: {cv_time}")

    """Modified remove diag functions for 1000 iterations (seconds)
    np_time: 1.2462092999994638
    o_time: 1.1972437940057716
    cv_time: 1.564783494999574"""


def test_remove_vert(img):
    @njit(cache=True)
    def _remove_horiz(remd: MatLike) -> MatLike:
        """Remove horizontal line

        Args:
            img (MatLike): Image (h, w, c)

        Returns:
            MatLike: Image with horizontal line removed
        """
        h, w, _ = remd.shape
        centre = h // 2

        for col in range(1, w - 1):
            remd[centre, col] = med_of(neighbours(remd, (centre, col)))

        return remd

    @njit(cache=True)
    def remove_horiz(img: MatLike) -> MatLike:
        """Remove horizontal line if Height is uneven

        Args:
            img (MatLike): _description_

        Returns:
            MatLike: _description_
        """

        h, _, _ = img.shape

        if h % 2 == 0:  # no line
            return img
        else:
            return _remove_horiz(img)

    @njit(cache=True)
    def remove_vert(img: MatLike) -> MatLike:
        img = img.transpose(1, 0, 2)
        remove_horiz(img)
        return img.transpose(1, 0, 2)

    @njit(cache=True)
    def _remove_vert(remd: MatLike) -> MatLike:
        h, w, _ = remd.shape
        centre = w // 2

        for row in range(1, h - 1):
            remd[row, centre] = med_of(neighbours(remd, (row, centre)))

        return remd

    @njit(cache=True)
    def remove_vert_n(img: MatLike) -> MatLike:
        _, w, _ = img.shape

        if w % 2 == 0:  # no line
            return img
        else:
            return _remove_vert(img)

    # Create test image

    img_size = (1001, 1001, 3)  # Adjust size as needed
    test_img = np.random.randint(0, 256, img_size, dtype=np.uint8)
    numiter = 1000

    # Warm up numba JIT
    test_copy = test_img.copy()
    remove_vert(test_copy)
    remove_vert_n(test_copy)
    remove_vert(img)
    remove_vert_n(img)

    vert_time = timeit.timeit(lambda: remove_vert(test_copy), number=numiter)
    vert_n_time = timeit.timeit(lambda: remove_vert_n(test_copy), number=numiter)
    vert_time_i = timeit.timeit(lambda: remove_vert(img), number=numiter)
    vert_n_time_i = timeit.timeit(lambda: remove_vert_n(img), number=numiter)

    print()
    print(f"Modified remove diag functions for {numiter} iterations (seconds)")
    print(f"vert_time: {vert_time} img.shape = {test_img.shape}")
    print(f"vert_n_time: {vert_n_time} img.shape = {test_img.shape}")
    print(f"vert_time: {vert_time_i} img.shape = {img.shape}")
    print(f"vert_n_time: {vert_n_time_i} img.shape = {img.shape}")

    """
    Modified remove diag functions for 1000 iterations (seconds)
    vert_time: 0.8617902740006684 img.shape = (1001, 1001, 3)
    vert_n_time: 0.8853045420037233 img.shape = (1001, 1001, 3)
    vert_time: 0.29470818799745757 img.shape = (375, 375, 3)
    vert_n_time: 0.2981681710007251 img.shape = (375, 375, 3)"""

    # minimal overhead, so going with remove_vert_n for code symmetry


def test_half_mirror(img):
    import mirror
    from old.mirror_1 import remove_horiz, remove_vert
    import lines

    def half_mirror2(img: MatLike, side: str):
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

        b, g, r = cv2.split(img)
        h, w = b.shape
        snum = side_codes[side]
        diagonals = lines.make_lines(h, w)
        b = mirror._project_1chan(b, snum, diagonals)
        g = mirror._project_1chan(g, snum, diagonals)
        r = mirror._project_1chan(r, snum, diagonals)
        img = cv2.merge((b, g, r), dst=img)
        return img

    def half_mirror(img: MatLike, side: str, disp=False):
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
        bl = blackout(img, side_codes[side])
        ln = side[1]
        mr = mirror.mirror(bl, line_codes[ln])

        # if disp:
        #     cv2.imshow("Square", img)
        #     cv2.waitKey(0)
        #     cv2.imshow("Blackout", bl)
        #     cv2.waitKey(0)
        #     cv2.imshow("Mirror", mr)
        #     cv2.waitKey(0)
        w = cv2.addWeighted(bl, 1.0, mr, 1.0, 0)
        if ln == "p":
            w = remove_diag_p(w)
        elif ln == "n":
            w = remove_diag_n(w)
        elif ln == "v":
            w = remove_horiz(w)
        elif ln == "h":
            w = remove_vert(w)
        return w

    # Create test image

    def run_all(func, img):
        side_codes = ["tv", "bv", "lh", "rh", "bp", "tp", "bn", "tn"]
        for code in side_codes:
            _ = func(img, code)

    img_size = (1001, 1001, 3)  # Adjust size as needed
    test_img = np.random.randint(0, 256, img_size, dtype=np.uint8)
    numiter = 1000 // 8

    # warm um NJIT and see some results
    test_copy = test_img.copy()
    side_codes = ["tv", "bv", "lh", "rh", "bp", "tp", "bn", "tn"]

    start = time.perf_counter()
    for code in side_codes:
        hm2 = half_mirror2(test_img, code)
        hm = half_mirror(test_copy, code)
        print(f"results the same for code {code}: {np.allclose(hm, hm2)}")
    elapsed = time.perf_counter() - start
    print()
    print(f"Warming up time: {elapsed:.6f} seconds")

    start = time.perf_counter()
    hm_time = timeit.timeit(lambda: run_all(half_mirror, test_copy), number=numiter)
    hm2_time = timeit.timeit(lambda: run_all(half_mirror2, test_img), number=numiter)
    elapsed = time.perf_counter() - start
    print(f"Warmed up time (big): {elapsed:.6f} seconds")

    start = time.perf_counter()
    hm_time_s = timeit.timeit(lambda: run_all(half_mirror, img), number=numiter)
    hm2_time_s = timeit.timeit(lambda: run_all(half_mirror2, img), number=numiter)
    elapsed = time.perf_counter() - start
    print(f"Warmed up time (small): {elapsed:.6f} seconds")

    print()
    print(f"Running half_mirror variants for each side (8) for {numiter} iterations ")
    print(f"Half mirror time: {hm_time} (seconds) Image size: {test_img.shape}")
    print(f"Half mirror 2 time: {hm2_time} (seconds) Image size: {test_img.shape}")
    print(f"Half mirror (small) time: {hm_time_s} (seconds) Image size: {img.shape}")
    print(f"Half mirror 2 (small) time: {hm2_time_s} (seconds) Image size: {img.shape}")

    """
    results the same for code tv: False
    results the same for code bv: False
    results the same for code lh: False
    results the same for code rh: False
    results the same for code bp: False
    results the same for code tp: False
    results the same for code bn: False
    results the same for code tn: False

    Warming up time: 0.537394 seconds
    Warmed up time (big): 8.739532 seconds
    Warmed up time (small): 1.078817 seconds

    Running half_mirror variants for each side (8) for 125 iterations 
    Half mirror time: 6.4294772959983675 (seconds) Image size: (1001, 1001, 3)
    Half mirror 2 time: 2.309689514004276 (seconds) Image size: (1001, 1001, 3)
    Half mirror (small) time: 0.8429876799928024 (seconds) Image size: (375, 375, 3)
    Half mirror 2 (small) time: 0.23543725500348955 (seconds) Image size: (375, 375, 3)
    """


def main():
    from test import test_crop

    img = test_crop()

    # test_remove_diag2()
    # test_remove_vert(img)
    test_half_mirror(img)


if __name__ == "__main__":
    main()
