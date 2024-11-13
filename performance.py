#!/usr/bin/env python3

import timeit
import cv2
import numpy as np
#local imports
from mat import mir_p, mir_n
from mat_original import mir_p_o, mir_n_o
from mirror import crop_square, blackout, mirror, remove_diag_n, remove_diag_p, make_diag_n, make_diag_p
from mirror_original import blackout_o, blackout_i, blackout_m, mirror_o, remove_diag_n_o, remove_diag_p_o

def test_mirs_img(img, numiter=100):
  
  #original functions
  o_mir_p = mir_p_o(img)
  o_mir_n = mir_n_o(img)
  print(f"Original mir functions for {numiter} iterations (seconds)")
  cv2.imshow('Original mir_p', o_mir_p)
  cv2.imshow('Original mir_n', o_mir_n)
  cv2.waitKey(0)
  o_mir_p_time = timeit.timeit(lambda: mir_p_o(img), number=numiter)
  o_mir_n_time = timeit.timeit(lambda: mir_n_o(img), number=numiter)
  print(f"Original mir_p time: {o_mir_p_time}")
  print(f"Original mir_n time: {o_mir_n_time}")
  
  #optimized functions
  mir_img = mir_p(img)
  mir2_img = mir_n(img)
  print()
  print(f"Modified mir functions for {numiter} iterations (seconds)")
  cv2.imshow('mir_p', mir_img)
  cv2.imshow('mir_n', mir2_img)
  cv2.waitKey(0)
  mir_time = timeit.timeit(lambda: mir_p(img), number=numiter)
  mir2_time = timeit.timeit(lambda: mir_n(img), number=numiter)
  print(f"mir_p time: {mir_time}")
  print(f"mir_p time: {mir2_time}")
  '''
  Original mir functions for 100 iterations (seconds)
  Original mir_p time: 0.00047601999904145487
  Original mir_n time: 2.157500057364814e-05
  
  Modified mir functions for 100 iterations (seconds)
  mir_p time: 9.301099998992868e-05
  mir_p time: 8.293099745060317e-05'''

def test_blackout_img(img, numiter=100):
  #original function
  o_blackouttv = blackout_o(img, 'tv')
  o_blackoutbv = blackout_o(img, 'bv')
  o_blackoutlh = blackout_o(img, 'lh')
  o_blackoutrh = blackout_o(img, 'rh')
  o_blackoutbp = blackout_o(img, 'bp')
  o_blackouttp = blackout_o(img, 'tp')
  o_blackoutbn = blackout_o(img, 'bn')
  o_blackouttn = blackout_o(img, 'tn')
  
  cv2.imshow('Original blackout tv', o_blackouttv)
  cv2.waitKey(0)
  cv2.imshow('Original blackout bv', o_blackoutbv)
  cv2.waitKey(0)
  cv2.imshow('Original blackout lh', o_blackoutlh)
  cv2.waitKey(0)
  cv2.imshow('Original blackout rh', o_blackoutrh)
  cv2.waitKey(0)
  cv2.imshow('Original blackout bp', o_blackoutbp)
  cv2.waitKey(0)
  cv2.imshow('Original blackout tp', o_blackouttp)
  cv2.waitKey(0)
  cv2.imshow('Original blackout bn', o_blackoutbn)
  cv2.waitKey(0)
  cv2.imshow('Original blackout tn', o_blackouttn)  
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  o_blackout_timetv = timeit.timeit(lambda: blackout_o(img, 'tv'), number=numiter)
  o_blackout_timebv = timeit.timeit(lambda: blackout_o(img, 'bv'), number=numiter)
  o_blackout_timelh = timeit.timeit(lambda: blackout_o(img, 'lh'), number=numiter)
  o_blackout_timerh = timeit.timeit(lambda: blackout_o(img, 'rh'), number=numiter)
  o_blackout_timebp = timeit.timeit(lambda: blackout_o(img, 'bp'), number=numiter)
  o_blackout_timetp = timeit.timeit(lambda: blackout_o(img, 'tp'), number=numiter)
  o_blackout_timebn = timeit.timeit(lambda: blackout_o(img, 'bn'), number=numiter)
  o_blackout_timetn = timeit.timeit(lambda: blackout_o(img, 'tn'), number=numiter)
  
  print(f"Original functions for {numiter} iterations (seconds)")
  print(f"Original blackout time tv: {o_blackout_timetv}")
  print(f"Original blackout time bv: {o_blackout_timebv}")
  print(f"Original blackout time lh: {o_blackout_timelh}")
  print(f"Original blackout time rh: {o_blackout_timerh}")
  print(f"Original blackout time bp: {o_blackout_timebp}")
  print(f"Original blackout time tp: {o_blackout_timetp}")
  print(f"Original blackout time bn: {o_blackout_timebn}")
  print(f"Original blackout time tn: {o_blackout_timetn}")
  
  #optimized function
  tv, bv, lh, rh, bp, tp, bn, tn = 0, 1, 2, 3, 4, 5, 6, 7# numba is better with numbers
  blackout_tv = blackout(img, tv)
  blackout_bv = blackout(img, bv)
  blackout_lh = blackout(img, lh)
  blackout_rh = blackout(img, rh)
  blackout_bp = blackout(img, bp)
  blackout_tp = blackout(img, tp)
  blackout_bn = blackout(img, bn)
  blackout_tn = blackout(img, tn)
  
  cv2.imshow('blackout v2 tv', blackout_tv)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 bv', blackout_bv)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 lh', blackout_lh)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 rh', blackout_rh)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 bp', blackout_bp)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 tp', blackout_tp)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 bn', blackout_bn)
  cv2.waitKey(0)
  cv2.imshow('blackout v2 tn', blackout_tn)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()

  blackout_time_tv = timeit.timeit(lambda: blackout(img, tv), number=numiter)
  blackout_time_bv = timeit.timeit(lambda: blackout(img, bv), number=numiter)
  blackout_time_lh = timeit.timeit(lambda: blackout(img, lh), number=numiter)
  blackout_time_rh = timeit.timeit(lambda: blackout(img, rh), number=numiter)
  blackout_time_bp = timeit.timeit(lambda: blackout(img, bp), number=numiter)
  blackout_time_tp = timeit.timeit(lambda: blackout(img, tp), number=numiter)
  blackout_time_bn = timeit.timeit(lambda: blackout(img, bn), number=numiter)
  blackout_time_tn = timeit.timeit(lambda: blackout(img, tn), number=numiter)
  
  print()
  print(f"Jit compiled functions for {numiter} iterations (seconds)")
  print(f"blackout time tv: {blackout_time_tv}")
  print(f"blackout time bv: {blackout_time_bv}")
  print(f"blackout time lh: {blackout_time_lh}")
  print(f"blackout time rh: {blackout_time_rh}")
  print(f"blackout time bp: {blackout_time_bp}")
  print(f"blackout time tp: {blackout_time_tp}")
  print(f"blackout time bn: {blackout_time_bn}")
  print(f"blackout time tn: {blackout_time_tn}")
  '''
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
  blackout time tn: 0.451004742000805'''

  
def test_modified_blackouts(img, numiter=100):
  cp, cpi, cpm = img.copy(), img.copy(), img.copy()
  
  #first partially jit compiled
  tv, bv, lh, rh, bp, tp, bn, tn = 0, 1, 2, 3, 4, 5, 6, 7# numba is better with numbers
  blackouttv0 = blackout(cp, tv)
  blackoutbv0 = blackout(cp, bv)
  blackoutlh0 = blackout(cp, lh)
  blackoutrh0 = blackout(cp, rh)
  blackoutbp0 = blackout(cp, bp)
  blackouttp0 = blackout(cp, tp)
  blackoutbn0 = blackout(cp, bn)
  blackouttn0 = blackout(cp, tn)
  
  cv2.imshow('blackout_v2_tv', blackouttv0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_bv', blackoutbv0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_lh', blackoutlh0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_rh', blackoutrh0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_bp', blackoutbp0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_tp', blackouttp0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_bn', blackoutbn0)
  cv2.waitKey(0)
  cv2.imshow('blackout_v2_tn', blackouttn0)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  blackout_time_tv0 = timeit.timeit(lambda: blackout(cp, tv), number=numiter)
  blackout_time_bv0 = timeit.timeit(lambda: blackout(cp, bv), number=numiter)
  blackout_time_lh0 = timeit.timeit(lambda: blackout(cp, lh), number=numiter)
  blackout_time_rh0 = timeit.timeit(lambda: blackout(cp, rh), number=numiter)
  blackout_time_bp0 = timeit.timeit(lambda: blackout(cp, bp), number=numiter)
  blackout_time_tp0 = timeit.timeit(lambda: blackout(cp, tp), number=numiter)
  blackout_time_bn0 = timeit.timeit(lambda: blackout(cp, bn), number=numiter)
  blackout_time_tn0 = timeit.timeit(lambda: blackout(cp, tn), number=numiter)
  
  print()
  print(f"blackoutv2 (partially jit compiled) for {numiter} iterations (seconds)")
  print(f"blackout time tv: {blackout_time_tv0}")
  print(f"blackout time bv: {blackout_time_bv0}")
  print(f"blackout time lh: {blackout_time_lh0}")
  print(f"blackout time rh: {blackout_time_rh0}")
  print(f"blackout time bp: {blackout_time_bp0}")
  print(f"blackout time tp: {blackout_time_tp0}")
  print(f"blackout time bn: {blackout_time_bn0}")
  print(f"blackout time tn: {blackout_time_tn0}")
  
  #fully jit compiled, mutates the image
  blackouttv_m = blackout_m(cpi, tv)
  blackoutbv_m = blackout_m(cpi, bv)
  blackoutlh_m = blackout_m(cpi, lh)
  blackoutrh_m = blackout_m(cpi, rh)
  blackoutbp_m = blackout_m(cpi, bp)
  blackouttp_m = blackout_m(cpi, tp)
  blackoutbn_m = blackout_m(cpi, bn)
  blackouttn_m = blackout_m(cpi, tn)
  
  cv2.imshow('blackout_m_tv', blackouttv_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_bv', blackoutbv_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_lh', blackoutlh_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_rh', blackoutrh_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_bp', blackoutbp_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_tp', blackouttp_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_bn', blackoutbn_m)
  cv2.waitKey(0)
  cv2.imshow('blackout_m_tn', blackouttn_m)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  blackout_time_tv_m = timeit.timeit(lambda: blackout_m(cpi, tv), number=numiter)
  blackout_time_bv_m = timeit.timeit(lambda: blackout_m(cpi, bv), number=numiter)
  blackout_time_lh_m = timeit.timeit(lambda: blackout_m(cpi, lh), number=numiter)
  blackout_time_rh_m = timeit.timeit(lambda: blackout_m(cpi, rh), number=numiter)
  blackout_time_bp_m = timeit.timeit(lambda: blackout_m(cpi, bp), number=numiter)
  blackout_time_tp_m = timeit.timeit(lambda: blackout_m(cpi, tp), number=numiter)
  blackout_time_bn_m = timeit.timeit(lambda: blackout_m(cpi, bn), number=numiter)
  blackout_time_tn_m = timeit.timeit(lambda: blackout_m(cpi, tn), number=numiter)
  
  print()
  print(f"blackoutv3_m (fully jit, mutates) for {numiter} iterations (seconds)")
  print(f"blackout time tv: {blackout_time_tv_m}")
  print(f"blackout time bv: {blackout_time_bv_m}")
  print(f"blackout time lh: {blackout_time_lh_m}")
  print(f"blackout time rh: {blackout_time_rh_m}")
  print(f"blackout time bp: {blackout_time_bp_m}")
  print(f"blackout time tp: {blackout_time_tp_m}")
  print(f"blackout time bn: {blackout_time_bn_m}")
  print(f"blackout time tn: {blackout_time_tn_m}")
  
  #fully jit compiled, immutable
  blackouttv_i = blackout_i(cpm, tv)
  blackoutbv_i = blackout_i(cpm, bv)
  blackoutlh_i = blackout_i(cpm, lh)
  blackoutrh_i = blackout_i(cpm, rh)
  blackoutbp_i = blackout_i(cpm, bp)
  blackouttp_i = blackout_i(cpm, tp)
  blackoutbn_i = blackout_i(cpm, bn)
  blackouttn_i = blackout_i(cpm, tn)
  
  cv2.imshow('blackout_i_tv', blackouttv_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_bv', blackoutbv_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_lh', blackoutlh_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_rh', blackoutrh_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_bp', blackoutbp_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_tp', blackouttp_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_bn', blackoutbn_i)
  cv2.waitKey(0)
  cv2.imshow('blackout_i_tn', blackouttn_i)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  blackout_time_tv_i = timeit.timeit(lambda: blackout_i(cpm, tv), number=numiter)
  blackout_time_bv_i = timeit.timeit(lambda: blackout_i(cpm, bv), number=numiter)
  blackout_time_lh_i = timeit.timeit(lambda: blackout_i(cpm, lh), number=numiter)
  blackout_time_rh_i = timeit.timeit(lambda: blackout_i(cpm, rh), number=numiter)
  blackout_time_bp_i = timeit.timeit(lambda: blackout_i(cpm, bp), number=numiter)
  blackout_time_tp_i = timeit.timeit(lambda: blackout_i(cpm, tp), number=numiter)
  blackout_time_bn_i = timeit.timeit(lambda: blackout_i(cpm, bn), number=numiter)
  blackout_time_tn_i = timeit.timeit(lambda: blackout_i(cpm, tn), number=numiter)
  
  print()
  print(f"blackoutv3_i (fully jit, immutable) for {numiter} iterations (seconds)")
  print(f"blackout time tv: {blackout_time_tv_i}")
  print(f"blackout time bv: {blackout_time_bv_i}")
  print(f"blackout time lh: {blackout_time_lh_i}")
  print(f"blackout time rh: {blackout_time_rh_i}")
  print(f"blackout time bp: {blackout_time_bp_i}")
  print(f"blackout time tp: {blackout_time_tp_i}")
  print(f"blackout time bn: {blackout_time_bn_i}")
  print(f"blackout time tn: {blackout_time_tn_i}")
  
  '''
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
  blackout time tn: 0.6046839369992085'''
  
def test_mirror(img, numiter=100, disp=False):
  v, h, p, n = 0, 1, 2, 3
  #original functions
  o_mirrorv = mirror_o(img, v)
  o_mirrorh = mirror_o(img, h)
  o_mirrorp = mirror_o(img, p)
  o_mirrorn = mirror_o(img, n)
  
  if disp:
    cv2.imshow('Original mirror v', o_mirrorv)
    cv2.waitKey(0)
    cv2.imshow('Original mirror h', o_mirrorh)
    cv2.waitKey(0)
    cv2.imshow('Original mirror p', o_mirrorp)
    cv2.waitKey(0)
    cv2.imshow('Original mirror n', o_mirrorn)
    cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  o_mirror_timev = timeit.timeit(lambda: mirror_o(img, v), number=numiter)
  o_mirror_timeh = timeit.timeit(lambda: mirror_o(img, h), number=numiter)
  o_mirror_timep = timeit.timeit(lambda: mirror_o(img, p), number=numiter)
  o_mirror_timen = timeit.timeit(lambda: mirror_o(img, n), number=numiter)
  
  print()
  print(f"Original mirror functions for {numiter} iterations (seconds)")
  print(f"Original mirror time v: {o_mirror_timev}")
  print(f"Original mirror time h: {o_mirror_timeh}")
  print(f"Original mirror time p: {o_mirror_timep}")
  print(f"Original mirror time n: {o_mirror_timen}")
  
  #optimized functions
  mirrorv = mirror(img, v)
  mirrorh = mirror(img, h)
  mirrorp = mirror(img, p)
  mirrorn = mirror(img, n)

  if disp:
    cv2.imshow('mirror v', mirrorv)
    cv2.waitKey(0)
    cv2.imshow('mirror h', mirrorh)
    cv2.waitKey(0)
    cv2.imshow('mirror p', mirrorp)
    cv2.waitKey(0)
    cv2.imshow('mirror n', mirrorn)
    cv2.waitKey(0)
  
  cv2.destroyAllWindows()
  
  mirror_timev = timeit.timeit(lambda: mirror(img, v), number=numiter)
  mirror_timeh = timeit.timeit(lambda: mirror(img, h), number=numiter)
  mirror_timep = timeit.timeit(lambda: mirror(img, p), number=numiter)
  mirror_timen = timeit.timeit(lambda: mirror(img, n), number=numiter)
  
  print()
  print(f"Modified mirror functions for {numiter} iterations (seconds)")
  print(f"mirror time v: {mirror_timev}")
  print(f"mirror time h: {mirror_timeh}")
  print(f"mirror time p: {mirror_timep}")
  print(f"mirror time n: {mirror_timen}")
  
  '''
  Original mirror functions for 100 iterations (seconds)
  Original mirror time v: 0.07256410300033167
  Original mirror time h: 0.06661735300440341
  Original mirror time p: 0.488386511002318
  Original mirror time n: 0.4426036499935435

  Modified mirror functions for 100 iterations (seconds)
  mirror time v: 0.043370286002755165
  mirror time h: 0.06154745399544481
  mirror time p: 0.4835698019960546
  mirror time n: 0.4653278540063184'''
  
def test_remove_diag(img, numiter=100):
  #make images
  diagp = make_diag_p(img)
  diagn = make_diag_n(img)
  
  cv2.imshow("Diag p", diagp)
  cv2.waitKey(0)
  cv2.imshow("Diag n", diagn)
  cv2.waitKey(0)
  
  #original functions
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
  
  #optimized functions
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
  
  '''
  Original remove diag functions for 100 iterations (seconds)
  Original remove diag time p: 8.11424048399931
  Original remove diag time n: 4.24769181100055

  Modified remove diag functions for 100 iterations (seconds)
  Modified remove diag time p: 0.16073157700157026
  Modified remove diag time n: 0.1617420470029174
  '''
  
def main():
  img = cv2.imread('src_imgs/landscape.jpg')
  img = crop_square(img)
  cv2.imshow('Original Image', img)
  cv2.waitKey(0)
  # print("testing mirs")
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # test_mirs_img(gray)
  # print()
  # print("---------------------------------------------------------------")
  # print()
  # print("testing blackouts")
  # test_blackout_img(img)
  # print()
  # print("Second iteration of blackouts")
  # test_modified_blackouts(img)
  # print()
  # print("---------------------------------------------------------------")
  # print()
  # print("testing mirror")
  # test_mirror(img)
  # print()
  # print("---------------------------------------------------------------")
  # print()
  print("Testing remove diag")
  test_remove_diag(img)
  

if __name__ == "__main__":
  main()