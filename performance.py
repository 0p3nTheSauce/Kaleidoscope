#!/usr/bin/env python3
from cv2.typing import MatLike
import timeit
import time
import cv2
import numpy as np

# local imports
from mat import mir_p, mir_n
from old.mat_0 import mir_p_o, mir_n_o
from old.mirror_1 import blackout, remove_diag_n, remove_diag_p



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
