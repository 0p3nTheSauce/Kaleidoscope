import cv2
import numpy as np
import sys
import statistics

from mat_original import mir_p_o, mir_n_o
from mirror import crop_square

def mirror_o(img, line):
  if line == 'v':
    return cv2.flip(img, 0)
  elif line == 'h':
    return cv2.flip(img, 1)
  elif line == 'p': #positive incline diagonal
    b, g, r = cv2.split(img)
    b = mir_p_o(b)
    g = mir_p_o(g)
    r = mir_p_o(r)
    return cv2.merge((b, g, r))
  elif line == 'n':#negative incline diagonal
    b, g, r = cv2.split(img)
    b = mir_n_o(b)
    g = mir_n_o(g)
    r = mir_n_o(r)
    return cv2.merge((b, g, r))
  else:
    print("Invalid line in mirror")
    sys.exit(1)
    
def blackout_o(img, line):
  #blacks out half an image
  # sqr = crop_square(img)
  h, w, _ = img.shape
  bl = img.copy()
  if line == 'tv':
    bl[h // 2:, :, :] = (0, 0, 0)
    return bl
  elif line == 'bv':
    bl[:h // 2, :, :] = (0, 0, 0)
    return bl
  elif line == 'lh':
    bl[:, w // 2:, :] = (0, 0, 0)
    return bl
  elif line == 'rh':
    bl[:, :w // 2, :] = (0, 0, 0)
    return bl
  elif line == 'bp':  
    m, c = 1, 0
    o = 'b'
  elif line == 'tp':
    m, c, = 1, 0
    o = 't'
  elif line == 'bn':#this is bottom negative slope
    m, c = -1, h
    o = 't' #think of this as operator, no correlation to name because of inversion
  elif line == 'tn':
    m, c = -1, h
    o = 'b'
  else:
    print("Invalid line")
    sys.exit(1)
  c = h - c
  m = -m
  for y in range(h):
    for x in range(w):
      yl = m * x + c
      if o == 'b':
        if y <= yl:
          bl[x, y] = (0, 0, 0)
      else:
        if y >= yl:
          bl[x, y] = (0, 0, 0)  
  return bl

def med_of_o(k):
  #get median tuple in kernel
  #used to remove diagnols
  t = list(zip(*k))
  med = tuple(statistics.median(dim) for dim in t)
  return med

def neighbours_n_o(img, coords):
  #gets the neighbouring pixel values 
  #used by remove_diag_n
  x, y = coords
  l = [img[y-1, x-1], img[y-1, x], img[y-1, x+1],
       img[y, x-1], img[y, x], img[y, x+1],
       img[y+1, x-1], img[y+1, x], img[y+1, x+1]]
  return l
            
            
def neighbours_p_o(img, coords):
  #gets the neighbouring pixel values 
  #used by remove_diag_p
  x, y = coords
  l = [img[x-1, y-1], img[x-1, y], img[x-1, y+1],
       img[x, y-1], img[x, y], img[x, y+1],
       img[x+1, y-1], img[x+1, y], img[x+1, y+1]]
  return l 

def remove_diag_n_o(img, disp=False):
  #removes diagnol lines (negative gradient) by taking the median 
  # of the neighbouring pixels
  h, w, _ = img.shape
  cp = img.copy()
  for rows in range(1, h-1):
    for cols in range(1, w-1):
      if rows == cols:
        k = neighbours_n_o(img, (rows, cols))
        med = med_of_o(k)
        cp[rows, cols] = med
  if disp:
    cv2.imshow('no diags', cp)
    cv2.waitKey(0)
  return cp
  
def remove_diag_p_o(img, disp=False):
  #removes diagnol lines (positive gradient) by taking the median 
  # of the neighbouring pixels
  h, w, _ = img.shape
  cp = img.copy()
  for rows in range(1, h-1):
    for cols in range(1, w-1):
      if (h-rows-1) == cols:
        k = neighbours_p_o(img, (rows, cols))
        med = med_of_o(k)
        cp[rows, cols] = med
  if disp:
    cv2.imshow('no diags', cp)
    cv2.waitKey(0)
  return cp

def mirror_o(img, line):
  if line == 0: #vertical
    return cv2.flip(img, 0)
  elif line == 1: #horizontal
    return cv2.flip(img, 1)
  elif line == 2: #positive incline diagonal
    b, g, r = cv2.split(img)
    b = mir_p_o(b)
    g = mir_p_o(g)
    r = mir_p_o(r)
    return cv2.merge((b, g, r))
  elif line == 3:#negative incline diagonal
    b, g, r = cv2.split(img)
    b = mir_n_o(b)
    g = mir_n_o(g)
    r = mir_n_o(r)
    return cv2.merge((b, g, r))
  else:
    print("Invalid line in mirror")
    sys.exit(1)

def half_mirror_o(img, line, disp=False):
  #mirrors half the image onto the other half
  sqr = crop_square(img)
  bl = blackout_o(sqr, line)
  ml = line[1]
  mr = mirror_o(bl, ml)
   
  if disp:
    cv2.imshow('Square', sqr)
    cv2.waitKey(0)
    cv2.imshow('Blackout', bl)
    cv2.waitKey(0)
    cv2.imshow('Mirror', mr)
    cv2.waitKey(0)
  w = cv2.addWeighted(bl, 1.0, mr, 1.0, 0)
  if ml == 'p':
    w = remove_diag_p_o(w)
  elif ml == 'n':
    w = remove_diag_n_o(w)
  return w