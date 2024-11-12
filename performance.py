import timeit
import cv2
import numpy as np
#local imports
from mat import mir_p, mir_n
from mat_original import mir_p_o, mir_n_o
from mirror import crop_square, blackout
from mirror_original import blackout_o

def test_mirs_img(img):
  
  #original functions
  o_mir_p = mir_p_o(img)
  o_mir_n = mir_n_o(img)
  cv2.imshow('Original mir', o_mir_p)
  cv2.imshow('Original mir2', o_mir_n)
  cv2.waitKey(0)
  o_mir_p_time = timeit.timeit(lambda: mir_p_o(img), number=100)
  o_mir_n_time = timeit.timeit(lambda: mir_n_o(img), number=100)
  print(f"Original mir time: {o_mir_p_time}")
  print(f"Original mir2 time: {o_mir_n_time}")
  
  #optimized functions
  mir_img = mir_p(img)
  mir2_img = mir_n(img)
  cv2.imshow('mir', mir_img)
  cv2.imshow('mir2', mir2_img)
  cv2.waitKey(0)
  mir_time = timeit.timeit(lambda: mir_p(img), number=100)
  mir2_time = timeit.timeit(lambda: mir_n(img), number=100)
  print(f"mir time: {mir_time}")
  print(f"mir2 time: {mir2_time}")

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
  
  cv2.imshow('Original blackout', o_blackouttv)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackoutbv)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackoutlh)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackoutrh)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackoutbp)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackouttp)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackoutbn)
  cv2.waitKey(0)
  cv2.imshow('Original blackout', o_blackouttn)  
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
  
  cv2.imshow('blackout', blackout_tv)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_bv)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_lh)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_rh)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_bp)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_tp)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_bn)
  cv2.waitKey(0)
  cv2.imshow('blackout', blackout_tn)
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


def main():
  img = cv2.imread('src_imgs/landscape.jpg')
  img = crop_square(img)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imshow('Original Image', img)
  cv2.waitKey(0)
  #test_mirs_img(img)
  test_blackout_img(img, numiter=100)

  

if __name__ == "__main__":
  main()