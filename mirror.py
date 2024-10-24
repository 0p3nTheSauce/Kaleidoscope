#!/usr/bin/env python3
import cv2
import numpy as np
import sys
#local
from mat import mir_diag

# def mirror(img, line):
#   m, c = line
#   h, w, _ = img.shape
#   cp = img.copy()
#   for y in range(h):
#     for x in range(w):
#       yl = m * x + c
#       xl = round((y - c) / m)
#       if y > yl:
#         cp[y, x] = img[yl, xl]
#   return cp

# def mirror2(img, line):
#   m, c = line
#   sqr = crop_square(img)
#   h, w, _ = sqr.shape
#   cp = sqr.copy()
#   c = h - c
#   m = -m
#   for y in range(h):
#     for x in range(w):
#       yl = m * x + c
#       if y == yl:
#         cp[y, x] = (0, 0, 255)
#       elif y < yl:
#         cp[x, y] = sqr[y, x]
  
#   return cp



def crop_square(img):
  h, w, _ = img.shape
  if h > w:
    return img[:w, :]
  return img[:, :h]

# def mirror3(img, line):
#   m, c = line
#   sqr = crop_square(img)
#   h, w, _ = sqr.shape
#   cp = sqr.copy()
#   for y in range(h):
#     for x in range(w):
#       yl = m * x + c
#       xl = round((y - c) / m)
#       if y > yl:
#         cp[y, x] = (0, 0, 0)
#   flip = cv2.flip(cp, 1)
#   # cv2.imshow('Flip', flip)
#   # cv2.waitKey(0)
#   M = cv2.getRotationMatrix2D((w // 2, h // 2), 90, 1)
#   rot = cv2.warpAffine(flip, M, (w, h))
#   # cv2.imshow('Rotated', rot)
#   # cv2.waitKey(0)
  
#   merged = cv2.addWeighted(cp, 1.0, rot, 1.0, 0)
  
#   return merged

# def mirror4(img, line):
#   m, c = line
#   sqr = crop_square(img)
#   h, w, _ = sqr.shape
#   cp = sqr.copy()
#   for y in range(h):
#     for x in range(w):
#       yl = m * x + c
#       if y < yl:
#         cp[x, y] = sqr[y, x]
  
#   return cp


def mirror(img, line):
  if line == 'v':
    return cv2.flip(img, 0)
  elif line == 'h':
    return cv2.flip(img, 1)
  elif line == 'p': #positive incline diagonal
    return mir_diag(img, 1)
  elif line == 'n':
    return mir_diag(img, -1) #negative incline diagonal
  else:
    print("Invalid line")
    sys.exit(1)

def blackout(img, line):
  # sqr = crop_square(img)
  h, w, _ = img.shape
  bl = img.copy()
  if line == 'v':
    bl[:h // 2, :, :] = (0, 0, 0)
    return bl
  elif line == 'h':
    bl[:, :w // 2, :] = (0, 0, 0)
    return bl
  elif line == 'p':  
    m, c = 1, 0
  elif line == 'n':
    m, c = -1, h
  else:
    print("Invalid line")
    sys.exit(1)
  c = h - c
  m = -m
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      # if y == yl:
      #   bl[y, x] = (0, 0, 255)
      if y < yl:
        bl[x, y] = (0, 0, 0)
  return bl

def half_mirror(img, line):
  sqr = crop_square(img)
  cv2.imshow('Square', sqr)
  cv2.waitKey(0)
  bl = blackout(sqr, line)
  cv2.imshow('Blackout', bl)
  cv2.waitKey(0)
  mr = mirror(bl, line)
  cv2.imshow('Mirror', mr)
  cv2.waitKey(0)
  return cv2.addWeighted(bl, 1.0, mr, 1.0, 0)

  

def main():
  img = cv2.imread('me.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  
  hm = half_mirror(img, 'v')
  cv2.imshow('Half Mirror', hm)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
if __name__ == '__main__':
  main()