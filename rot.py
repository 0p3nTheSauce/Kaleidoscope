#!/usr/bin/env python3
import cv2
import numpy as np
import random

def rotate(img, angle):
  h, w, _ = img.shape
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(img, M, (w, h))
  return rotated
  
def spin_det(img):
  #Image deteriorates with mumtiple rotations
  for i in range(100):
    img = rotate(img, 10)
    cv2.imshow('Rotated', img)
    cv2.waitKey(50)
  cv2.destroyAllWindows()
  
def spin(img):
  for i in range(0, 1000, 10):
    rot = rotate(img,i)
    cv2.imshow('Rotated', rot)
    cv2.waitKey(50)
  cv2.destroyAllWindows()
  
def color_wheel(img):
  for i in range(0, 1000, 10):
    rot = rotate(img,i)
    chan = color_change2(rot)
    cv2.imshow('Rotated', chan)
    cv2.waitKey(50)
  cv2.destroyAllWindows()
  
def color_change(img):
  chan = random.randint(0, 2)
  plus_col = random.randint(-255, 255)
  blank = np.zeros_like(img)
  blank[:,:,chan] = img[:,:,chan]
  temp_channel = blank[:, :, chan].astype(np.int16)
  temp_channel = np.clip(temp_channel + plus_col, 0, 255)
  blank[:, :, chan] = temp_channel.astype(np.uint8)
  return blank

def sharpen(img, type='G', disp=False, nb=1, ns=1):
  blur = img.copy()
  for i in range(nb):
    if type == 'G':
      blur = cv2.GaussianBlur(blur, (5, 5), 2)
    if type == 'N':
      blur = cv2.fastNlMeansDenoisingColored(blur, None, 10, 10 ,7 ,21)
  diff = cv2.subtract(img, blur)
  sharp = img.copy()
  for i in range(ns):
    sharp = cv2.addWeighted(sharp, 1.0, diff, 1.0, 0)
  if disp:
    cv2.imshow('Pattern', blur)
    cv2.waitKey(0)
    cv2.imshow('Difference', diff)
    cv2.waitKey(0)
    cv2.imshow('Pattern', sharp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return sharp
    
def gen_pattern(img ,type='G', time=50, iter=2):
  for i in range(iter):
    img = sharpen(img, type, nb=2, ns=4)
    cv2.imshow('Pattern', img)
    cv2.waitKey(time)

def gen_pattern2(img ,type='G', iter=2):
  for i in range(iter):
    img = sharpen(img, type)
  return img
    
def color_change2(img):
  chan = random.randint(0, 2)
  color = np.random.randint(0, 256, img.shape, dtype=np.uint8)
  color[:, :, chan] = img[:, :, chan]
  return color
  
def color_wheel2(img):
  for i in range(0, 1000, 10):
    rot = rotate(img,i)
    chan = color_change2(rot)
    pat = gen_pattern2(chan)
    cv2.imshow('spinny', pat)
    key = cv2.waitKey(50)
    if key == 27:
      break 
  cv2.destroyAllWindows()
  

def disk(img):
  h, w, _ = img.shape
  center = (w//2, h//2)
  l = [h, w]
  radius = min(l)//2
  mask_img = np.zeros_like(img)
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  cv2.circle(mask ,center, radius, (255), thickness=-1)
  mask_img[mask == 255] = img[mask == 255]
  return mask_img
  
def spin_func(img, func, params=None, iter=1000, deg=10, time=50, outfolder=None):
  if outfolder is not None:
    write = True
  else:
    write = False
  for i in range(0, iter, deg):
    rot = rotate(img,i)
    if params is not None:
      rot = func(rot, *params)
    else:
      rot = func(rot)
    cv2.imshow('Rotated', rot)
    if write:
      cv2.imwrite(f'{outfolder}/{i}.jpg', rot)
    key = cv2.waitKey(time)
    if key == 27:
      break
  cv2.destroyAllWindows()
  
  
def main():
  img = cv2.imread('me.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)

  # gen_pattern(img)
  track = disk(img)
  cv2.imshow('Track', track)
  cv2.waitKey(0)
  #spin(img)
  gray = cv2.imread('me.jpg', 0)
  cv2.imshow('gray', gray)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # color_wheel(img)
  color_wheel2(track)
  print()

if __name__ == '__main__':
  main()