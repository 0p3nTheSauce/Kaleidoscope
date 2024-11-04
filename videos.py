import cv2
import os
import subprocess
import sys

def images_to_video(path, fr = 30.0, ): 
  
  #.mp4 produced not compatible with whatsApp
  
  # out_path = f'{path}.avi'
  out_path = f'{path}.mp4'
  
  images = [img for img in os.listdir(path) if img.endswith('.jpg')]
  images.sort(key=lambda x: int(x.split('.')[0]))
  frame = cv2.imread(os.path.join(path, images[0]))
  h, w, c = frame.shape
  size = (w, h)
  
  # fourcc = cv2.VideoWriter_fourcc(*'XVID')
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
  writer = cv2.VideoWriter(out_path, fourcc, fr, size)
  
  for img in images:
    img_path = os.path.join(path, img)
    frame = cv2.imread(img_path)
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
      print(f"Video successfully overwritten at {original_file}")
  except subprocess.CalledProcessError as e:
      print("An error occurred while processing the video:", e)
      # Clean up temp file if there was an error
      if os.path.exists(temp_file):
          os.remove(temp_file)
          
def makeVideo(name, fr=30):
  images_to_video(name,fr)
  convert_mp4(name)
  
def main():
  if len(sys.argv) == 0:
    print("Usage: videos.py <input_directory>")
    print("Example: videos.py illusion")
    sys.exit(1)
  else:
    makeVideo(sys.argv[1])
  
if __name__ == "__main__":
  main()
  
  
  