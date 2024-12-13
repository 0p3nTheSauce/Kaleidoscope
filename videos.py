#!/usr/bin/env python3


import cv2
import os
import subprocess
import sys

def images_to_video(path, fr = 30.0, newSize=None ): 
  
  #.mp4 produced not compatible with whatsApp
  # newSize should be (w, h)
  
  # out_path = f'{path}.avi'
  out_path = f'{path}.mp4'
  
  images = [img for img in os.listdir(path) if img.endswith('.jpg')]
  images.sort(key=lambda x: int(x.split('.')[0]))
  
  resz = False
  if newSize is not None:
    size = newSize
    w, h = newSize
    resz = True
  else:
    frame = cv2.imread(os.path.join(path, images[0]))
    h, w, c = frame.shape
    size = (w, h) #this might cause confusion being opposite to openCV
  
  # fourcc = cv2.VideoWriter_fourcc(*'XVID')
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
  writer = cv2.VideoWriter(out_path, fourcc, fr, size)
  
  for img in images:
    img_path = os.path.join(path, img)
    frame = cv2.imread(img_path)
    if resz:
      frame = cv2.resize(frame, newSize)
    # cv2.imshow('image', frame)
    # key = cv2.waitKey(30)
    # if key == 27:
    #   break
    writer.write(frame)
  
  writer.release()
  print(f'Video saved at {out_path}')
  
def convert_mp4(name):
  
  #re-encode the .mp4
  
  original_file = f"{name}.mp4"
  temp_file = f"{name}_temp.mp4"

  # FFMPEG command to re-encode to H.264 and AAC
  command = [
      "ffmpeg", "-i", original_file,
      "-vcodec", "libx264", "-acodec", "aac",
      "-strict", "-2", temp_file
  ]

  try:
      subprocess.run(command, check=True)
      os.replace(temp_file, original_file)  
      print(f"Video successfully written at {original_file}")
  except subprocess.CalledProcessError as e:
      print("An error occurred while processing the video:", e)
      # Clean up temp file if there was an error
      if os.path.exists(temp_file):
          os.remove(temp_file)
          
def makeVideo(name, fr=30, newSize=None):
  images_to_video(name,fr, newSize)
  convert_mp4(name)
  
def main():
  if len(sys.argv) == 2:
    print("Making video")
    makeVideo(sys.argv[1])  
  elif len(sys.argv) == 4:
    try:
        w = int(sys.argv[2])
        h = int(sys.argv[3])
    except ValueError:
        print("Conversion failed: not a valid integer")
    print("Making video")
    makeVideo(sys.argv[1], newSize=(w,h))
  else:
    print("Usage: videos.py <input_directory> [ width height ]")
    print("Example: videos.py illusion 500 500")
    print("If width and height are not provided, the original size is used")
    sys.exit(1)
    
  
if __name__ == "__main__":
  main()
  
  
  