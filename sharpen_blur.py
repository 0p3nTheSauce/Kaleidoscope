#!/usr/bin/env python3

import cv2
import glob
import numpy as np


def makevideo(path, fr = 10.0, ):
  out_path = f'{path}.avi'
  
  images = sorted(glob.glob(f"{path}/*.JPG"))
  frame = cv2.imread(images[0])
  h, w, c = frame.shape
  size = (w, h)
  
  fourcc = cv2.VideoWriter_fourcc(*'XVID') 
  writer = cv2.VideoWriter(out_path, fourcc, fr, size)  
  
  for img in images:
    frame = cv2.imread(img)
    writer.write(frame)
  
  writer.release()
  print(f'Video saved at {out_path}')
  
def slid(img):
  n_b, n_s = 5, 5
  output_path = 'slid'
  while n_b > 0:
    n_b, n_s, img = slad(n_b, n_s, img, output_path)
  cv2.destroyAllWindows()

def slad(n_b, n_s, img, output_path):
  for i in range(n_b):
    img = blur(img, 2)
  for i in range(n_s):
    # img = sharpen(img, 2)
    blurred = blur(img, 2)
    diff = cv2.subtract(img, blurred)
    sharpen = cv2.addWeighted(img, 1.0, diff, 2.0, 0)
    img = sharpen
    cv2.imshow(f'Images{n_b}{n_s}', diff)
    cv2.imwrite(f"{output_path}/Images{n_b}{n_s}.JPG", diff)
  n_b -= 1
  n_s += 1
  # name = (f"Images{n_b}{n_s}")
  # cv2.imshow(name, img)
  # cv2.imwrite(f"{output_path}/{name}.JPG", img)
  return n_b, n_s, img

def sharpen(img, degree, disp=False, type='G'):
  blurred = blur(img, degree, type)
  diff = cv2.subtract(img, blurred)
  if disp:
    cv2.imshow('Diff', diff)
    key = cv2.waitKey(0)
    if key == ord('s'):
      cv2.imwrite('diff.JPG', diff)
    cv2.destroyAllWindows()
    return img
  sharpen = cv2.addWeighted(img, 1.0, diff, 1.0, 0)
  return sharpen

def sharpie(img, degree, type='G'):
  blurred = blur(img, degree, type)
  diff = cv2.subtract(img, blurred)
  sharpen = cv2.addWeighted(img, 1.0, diff, 0.5, 0)
  return sharpen, diff

def blur(img, degree, type='G'):
  if type == 'G':
    blur = cv2.GaussianBlur(img, (5, 5), degree)
  elif type == 'M':
    blur = cv2.medianBlur(img, degree+4)
  elif type == 'B':
    blur = cv2.bilateralFilter(img, 9, 75, 75)
  elif type == 'N':
    blur = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
  elif type == 'T':
    result = np.empty_like(img)
    cv2.denoise_TVL1(img, result, 1.0, 30)
    blur = result
  return blur

def skode(img, degree=2, type='G'):
  count = 0
  while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(0)
    if key == 27: #Esc
      break
    elif key == 81: #left arrow
      img = blur(img, degree, type)
    elif key == 83: #right arrow
      img, diff = sharpie(img, degree, type)
      cv2.imwrite(f'skode4/frame_{count}.JPG', diff)
      count += 1
    elif key == 82:
      degree += 1
    elif key == 84: #down arrow
      degree -= 1
    elif key == ord('d'):
      img = sharpen(img, degree, True)
    elif key == ord('g'):
      type = 'G'
      print(type)
    elif key == ord('m'):
      type = 'M'
      print(type)
    elif key == ord('b'):
      type = 'B'
      print(type)
    elif key == ord('n'):
      type = 'N'
      print(type)
    elif key == ord('t'):
      type = 'T'
      print(type)
  cv2.destroyAllWindows()
  
def slide(img, degree=1, type='G'):
  while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(0)
    if key == 27: #Esc
      break
    elif key == 81: #left arrow
      img = blur(img, degree, type)
    elif key == 83: #right arrow
      img = sharpen(img, degree)
    elif key == 82:
      degree += 1
      print(degree)
    elif key == 84: #down arrow
      degree -= 1
      print(degree)
    elif key == ord('d'):
      img = sharpen(img, degree, True)
    elif key == ord('g'):
      type = 'G'
      print(type)
    elif key == ord('m'):
      type = 'M'
      print(type)
    elif key == ord('b'):
      type = 'B'
      print(type)
    elif key == ord('n'):
      type = 'N'
      print(type)
    elif key == ord('t'):
      type = 'T'
      print(type)
    
  cv2.destroyAllWindows()
  
  
def main():
  img = cv2.imread('grad.JPG', 1)
  resized = cv2.resize(img, fx=0.5, fy=0.5)
  img = resized
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  blur = cv2.GaussianBlur(img, (7, 7), 2)
  cv2.imshow('Blur', blur)
  cv2.waitKey(0)
  diff = cv2.subtract(img, blur)
  cv2.imshow('Diff', diff)
  cv2.waitKey(0)
  sharpen = cv2.addWeighted(img, 1.0, diff, 2.0, 0)
  cv2.imshow('Sharpen', sharpen)
  cv2.waitKey(0)
if __name__ == "__main__":
  img = cv2.imread('grad.JPG', 1)
  resized = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
  #slide(resized)
  #slid(resized)
  skode(resized)
  makevideo('skode4', fr=2)
  #main()
