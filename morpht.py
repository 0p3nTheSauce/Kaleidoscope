import cv2
import numpy as np

def diler():
  img = cv2.imread('me.jpg', cv2.IMREAD_GRAYSCALE)
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  
  _, thresh = cv2.threshold(img, 150, 200, cv2.THRESH_BINARY)
  cv2.imshow('Thresh', thresh)
  cv2.waitKey(0)
  
  img = thresh
  
  k = np.ones((5,5), np.uint8)
  erd = cv2.erode(img, k, iterations=1)
  cv2.imshow('Eroded', erd)
  cv2.waitKey(0)
  
  dil = cv2.dilate(img, k, iterations=1)
  cv2.imshow('Dilated', dil)
  cv2.waitKey(0)
  
  opn = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
  cv2.imshow('Opened', opn)
  cv2.waitKey(0)
  
  cv2.destroyAllWindows()

def main():
  img = cv2.imread('me.jpg')
  cv2.imshow('Original', img)
  cv2.waitKey(0)
  
  img2 = cv2.imread('mush.jpg')
  img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
  cv2.imshow('Mush', img2)
  cv2.waitKey(0)
  
  mask = np.zeros(img.shape[:2], np.uint8)
  # mask2 = np.ones(img.shape[:2], np.uint8)
  # mask2 = mask2 * 255
  
  cv2.circle(mask, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
  mask2 = cv2.bitwise_not(mask)
  
  
  
  result = cv2.bitwise_and(img, img, mask=mask)
  result2 = cv2.bitwise_and(img2, img2, mask=mask2)
  cv2.imshow('Masked', result)
  cv2.waitKey(0)
  cv2.imshow('Masked2', result2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  result3 = cv2.add(result, result2)
  cv2.imshow('Added', result3)
  cv2.waitKey(0)
  dump = cv2.addWeighted(img, 0.5, img2, 0.5, 0)
  cv2.imshow('Dump', dump)
  cv2.waitKey(0)

if __name__ == "__main__":
  main()