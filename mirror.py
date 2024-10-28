#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import random
import statistics
#local
from mat import mir, mir2
from rot import spin_func, disk

def crop_square(img):
  h, w, _ = img.shape
  if h > w:
    return img[:w, :]
  return img[:, :h]

def mirror(img, line):
  if line == 'v':
    return cv2.flip(img, 0)
  elif line == 'h':
    return cv2.flip(img, 1)
  elif line == 'p': #positive incline diagonal
    b, g, r = cv2.split(img)
    b = mir(b)
    g = mir(g)
    r = mir(r)
    return cv2.merge((b, g, r))
  elif line == 'n':#negative incline diagonal
    b, g, r = cv2.split(img)
    b = mir2(b)
    g = mir2(g)
    r = mir2(r)
    return cv2.merge((b, g, r))
  else:
    print("Invalid line in mirror")
    sys.exit(1)

def blackout(img, line, remove_diag=True):
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
      # if remove_diag and y == yl:
      #   (b, g, r) = bl[y, x]
      #   bl[y, x] = (0, 0, 0)
      #   print('removed diag')
      if o == 'b':
        if y <= yl:
          bl[x, y] = (0, 0, 0)
      else:
        if y >= yl:
          bl[x, y] = (0, 0, 0)  
  return bl


def med_of(k):
  #get median tuple in kernel
  t = list(zip(*k))
  med = tuple(statistics.median(dim) for dim in t)
  return med

def remove_diags(img, ml):
  #remove the diagnol lines that form in half mirror
  rmd = img.copy()
  if ml == 'p':
    m, c = 1, 0
  elif ml == 'n':
    m, c = -1, 0
  else:
    print("Invalid ml in remove diags")
    sys.exit(1)
  h, w, _ = img.shape
  for y in range(1, h-1):
    for x in range(1, w-1):
      if y == m * x + c:
        lb = [img[y-1, x-1], img[y-1, x], img[y-1, x+1], img[y, x-1], img[y, x+1], img[y+1, x-1], img[y+1, x], img[y+1, x+1]] #3x3 kernel
        rmd[y, x] = med_of(lb)
  return rmd
            


def half_mirror(img, line, disp=False):
  sqr = crop_square(img)
  bl = blackout(sqr, line)
  ml = line[1]
  mr = mirror(bl, ml)
   
  if disp:
    cv2.imshow('Square', sqr)
    cv2.waitKey(0)
    cv2.imshow('Blackout', bl)
    cv2.waitKey(0)
    cv2.imshow('Mirror', mr)
    cv2.waitKey(0)
  w = cv2.addWeighted(bl, 1.0, mr, 1.0, 0)
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
  
def rand_color():
  b = random.randint(0, 255)
  g = random.randint(0, 255)
  r = random.randint(0, 255)
  color = (b, g, r)
  return color
  
def set_color(gray, color):
  b, g, r = color
  bm = gray * b
  gm = gray * g
  rm = gray * r
  return cv2.merge((bm, gm, rm))
  
def edgey(img, time=50):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 100, 200)
  colour = ((255, 255, 255))
  for i in range(0, 255, 5):
    for j in range(0, 255, 5):
      edges = cv2.Canny(gray, i, j)
      edges = set_color(edges, colour)
      edges = text_img(edges, f'lower: {i}, upper: {j}', color=colour)
      cv2.imshow('Edges', edges)
      key = cv2.waitKey(time)
      if key == 27:
        cv2.destroyAllWindows()
        return
    colour = rand_color()
  cv2.destroyAllWindows()
  
def edgey_sing(img, disp=False, text=False):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ns = [random.randint(0, 255) for i in range(2)]
  ns = sorted(ns)
  lb = ns[0]
  ub = ns[1]
  # lb = random.randint(0, 100)
  # ub = random.randint(100, 255)
  
  edges = cv2.Canny(gray, lb, ub)
  colour = rand_color()
  edges = set_color(edges, colour)
  if text:
    edges = text_img(edges, f'lower: {lb}, upper: {ub}', color=colour)
  if disp:
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
  return edges
  


def text_img(img, text, disp=False ,color=(0, 255, 0)):
  image = img.copy()
  position = (50, 50)  # Position (x, y)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  # color = (0, 255, 0)  # Text color (B, G, R)
  thickness = 2
  line_type = cv2.LINE_AA

  # Put the text on the image
  cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)

  if disp:
  # Display the image with text
    cv2.imshow("Image with Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
  return image




def main():
  img = cv2.imread('me.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)

  hm = half_mirror(img, 'tp', disp=True)
  cv2.imshow('Half Mirror', hm)
  cv2.waitKey(0)
  
  
  track = disk(img)
  cv2.imshow('Track', track)
  cv2.waitKey(0)
  
  #spin_func(track, edgey_sing, iter=1000, deg=1, time=20)
  
  #spin_func(img, multi_mirror) #very cool
  
  #edgey(img ,time=20)
  
  
  #hm = multi_mirror(img, disp=True)
  
  cv2.destroyAllWindows()
  #spin_mirror(img)
if __name__ == '__main__':
  main()