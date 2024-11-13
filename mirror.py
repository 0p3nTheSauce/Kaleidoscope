#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import sys
from numba import njit
#local
from mat import mir_p, mir_n
from rot import spin_func, disk
from videos import makeVideo

def crop_square(img):
  h, w, _ = img.shape
  if h > w:
    diff = h - w
    dif1 = diff//2
    dif2 = diff-dif1
    return img[dif1:h-dif2, :]
  diff = w - h
  dif1 = diff//2
  dif2 = diff-dif1
  return img[:, dif1:w-dif2]

def mirror(img, line):
  if line == 0: #vertical
    return cv2.flip(img, 0)
  elif line == 1: #horizontal
    return cv2.flip(img, 1)
  elif line == 2: #positive incline diagonal
    b, g, r = cv2.split(img)
    b = mir_p(b)
    g = mir_p(g)
    r = mir_p(r)
    return cv2.merge((b, g, r))
  elif line == 3:#negative incline diagonal
    b, g, r = cv2.split(img)
    b = mir_n(b)
    g = mir_n(g)
    r = mir_n(r)
    return cv2.merge((b, g, r))
  else:
    print("Invalid line in mirror")
    sys.exit(1)



@njit(cache=True)
def blackout_1chan(img, side):
  h, w = img.shape
  bl = img.copy()
  b, t = 1, 0
  match side:
    case 0: # top vertical
      bl[h // 2:, :] = 0
    case 1: # bottom vertical
      bl[:h // 2, :] = 0
    case 2: # left horizontal
      bl[:, w // 2:] = 0
    case 3: # right horizontal 
      bl[:, :w // 2] = 0
    case 4: # bottom positive slope
      m, c, o = 1, 0, b
    case 5: # top positive slope
      m, c, o = 1, 0, t
    case 6: # bottom negative slope
      m, c, o = -1, h, t
    case 7: # top negative slope
      m, c, o = -1, h, b
    case _:
      raise ValueError("Invalid line code")

  #handle diagonal lines
  if side >= 4:
    c = h - c
    m = -m
    for y in range(h):
      for x in range(w):
        yl = m * x + c
        if (o == b and y <= yl) or (o == t and y >= yl):
          bl[x, y] = 0
          
  return bl

def blackout(img, side):
  b, g, r = cv2.split(img)
  b = blackout_1chan(b, side)
  g = blackout_1chan(g, side)
  r = blackout_1chan(r, side)
  return cv2.merge((b, g, r))

def make_diag_n(img, color=(0, 255, 0), disp=False):
  #make diagnol line (negative gradient)
  h, w, _ = img.shape
  cp = img.copy()
  for rows in range(h):
    for columns in range(w):
      if rows == columns:
        cp[rows, columns] = color
  if disp:
    cv2.imshow('diag', cp)
    cv2.waitKey(0)
  return cp

def make_diag_p(img, color=(0,255,0), disp=False):
  #make diagnol line (negative gradient)
  h, w, _ = img.shape
  cp = img.copy()
  for rows in range(h):
    for columns in range(w):
      if (h-1-rows) == columns:
        cp[rows, columns] = color
  if disp:
    cv2.imshow('diag', cp)
    cv2.waitKey(0)
  return cp


@njit(cache=True)
def med_of(k):
    # Separate channels manually for each color dimension
    r = np.array([px[0] for px in k])
    g = np.array([px[1] for px in k])
    b = np.array([px[2] for px in k])
    
    # Find the median for each channel
    med_r = np.median(r)
    med_g = np.median(g)
    med_b = np.median(b)
    
    return (med_r, med_g, med_b)

@njit(cache=True)
def neighbours_n(img, coords):
  # Gets the neighbouring pixel values
  x, y = coords
  return [
    img[y-1, x-1], img[y-1, x], img[y-1, x+1],
    img[y, x-1], img[y, x], img[y, x+1],
    img[y+1, x-1], img[y+1, x], img[y+1, x+1]
  ]

@njit(cache=True)
def neighbours_p(img, coords):
  #gets the neighbouring pixel values 
  x, y = coords
  return [
    img[x-1, y-1], img[x-1, y], img[x-1, y+1],
    img[x, y-1], img[x, y], img[x, y+1],
    img[x+1, y-1], img[x+1, y], img[x+1, y+1]
  ]

@njit(cache=True)
def remove_diag_n(img):
  #removes diagnol lines (negative gradient) by taking the median 
  # of the neighbouring pixels
  h, w, _ = img.shape
  for rows in range(1, h-1):
    for cols in range(1, w-1):
      if rows == cols:
        k = neighbours_n(img, (rows, cols))
        med = med_of(k)
        img[rows, cols] = med
  return img

@njit(cache=True)
def remove_diag_p(img):
  #removes diagnol lines (positive gradient) by taking the median 
  # of the neighbouring pixels
  h, w, _ = img.shape
  for rows in range(1, h-1):
    for cols in range(1, w-1):
      if (h-rows-1) == cols:
        k = neighbours_p(img, (rows, cols))
        med = med_of(k)
        img[rows, cols] = med
  return img

def half_mirror(img, side, disp=False):
  #mirrors half the image onto the other half
  side_codes = {'tv': 0, 'bv': 1, 'lh': 2, 'rh': 3, 'bp': 4, 'tp': 5, 'bn': 6, 'tn': 7}
  line_codes = {'v': 0, 'h': 1, 'p': 2, 'n': 3}
  sqr = crop_square(img)
  bl = blackout(sqr, side_codes[side])
  ln = side[1]
  mr = mirror(bl, line_codes[ln])
   
  if disp:
    cv2.imshow('Square', sqr)
    cv2.waitKey(0)
    cv2.imshow('Blackout', bl)
    cv2.waitKey(0)
    cv2.imshow('Mirror', mr)
    cv2.waitKey(0)
  w = cv2.addWeighted(bl, 1.0, mr, 1.0, 0)
  if ln == 'p':
    w = remove_diag_p(w)
  elif ln == 'n':
    w = remove_diag_n(w)
  return w


def spin_mirror(img):
  cv2.imshow('Image', img)
  cv2.waitKey(50)
  cp = img.copy()
  mrs = ['lh', 'tp', 'tv', 'tn', 'rh', 'bp', 'bv', 'bn']
  for i in range(10):
    for l in mrs:
      hm = half_mirror(cp, l)
      cv2.imshow('Half Mirror', hm)
      key = cv2.waitKey(100)
      if key == 27:
        break
  cv2.destroyAllWindows()
    
def multi_mirror(img, mrs=['tn', 'tp', 'tv', 'rh'], disp=False):
  hm = img.copy()
  for l in mrs:
    hm = half_mirror(hm, l)
    if disp:
      cv2.imshow('Half Mirror', hm)
      cv2.waitKey(0)
  return hm
  
def main():
  if len(sys.argv) == 1:
    in_img = 'src_imgs/mush'
    out = 'mush'
  else:
    # out = sys.argv[1]
    print("this hasn't been fixed yet...")
    sys.exit(1)
  print(f'{in_img}.jpg')
  img = cv2.imread(f'{in_img}.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  img = crop_square(img)
  #img = cv2.resize(img, (500, 500))
  cv2.imshow('Crop squared: ', img)
  cv2.waitKey(0)

  #spin_func(track, edgey_sing, iter=1000, deg=1, time=20)
  #spin_func(img, multi_mirror, time=1, outfolder=out) #very cool
  #spin_func(img, multi_mirror, time=1, outfolder=out)
  spin_func(img, multi_mirror, time=1)
  #makeVideo(out)
  #edgey(img ,time=20)
  
  
  #hm = multi_mirror(img, disp=True)
  
  cv2.destroyAllWindows()
  #spin_mirror(img)
if __name__ == '__main__':
  main()