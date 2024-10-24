#!/usr/bin/env python3
import cv2
import numpy as np

def mirror(img, line):
  m, c = line
  h, w, _ = img.shape
  cp = img.copy()
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      xl = round((y - c) / m)
      if y > yl:
        cp[y, x] = img[yl, xl]
  return cp

def mirror2(img, line):
  m, c = line
  sqr = crop_square(img)
  h, w, _ = sqr.shape
  cp = sqr.copy()
  c = h - c
  m = -m
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      if y == yl:
        cp[y, x] = (0, 0, 255)
      elif y < yl:
        cp[x, y] = sqr[y, x]
  
  return cp



def crop_square(img):
  h, w, _ = img.shape
  if h > w:
    return img[:w, :]
  return img[:, :h]

def mirror3(img, line):
  m, c = line
  sqr = crop_square(img)
  h, w, _ = sqr.shape
  cp = sqr.copy()
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      xl = round((y - c) / m)
      if y > yl:
        cp[y, x] = (0, 0, 0)
  flip = cv2.flip(cp, 1)
  # cv2.imshow('Flip', flip)
  # cv2.waitKey(0)
  M = cv2.getRotationMatrix2D((w // 2, h // 2), 90, 1)
  rot = cv2.warpAffine(flip, M, (w, h))
  # cv2.imshow('Rotated', rot)
  # cv2.waitKey(0)
  
  merged = cv2.addWeighted(cp, 1.0, rot, 1.0, 0)
  
  return merged

def mirror4(img, line):
  m, c = line
  sqr = crop_square(img)
  h, w, _ = sqr.shape
  cp = sqr.copy()
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      if y < yl:
        cp[x, y] = sqr[y, x]
  
  return cp


    

def main():
  img = cv2.imread('me.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  sqr = crop_square(img)
  cv2.imshow('Square', sqr)
  cv2.waitKey(0)  
  print(sqr.shape)
  mir = mirror4(img, (0, 240))
  cv2.imshow('Mirror', mir)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  pass

if __name__ == '__main__':
  main()