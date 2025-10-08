#!/usr/bin/env python3
import cv2
from cv2.typing import MatLike
from typing import Callable, Optional
import os

def rotate(img: MatLike, angle: float) -> MatLike:
    """Shorthand for cv2.getRotationMatrix2D followed by cv2.warpAffine. Rotates
    in centre of image. 

    Args:
        img (MatLike): Image
        angle (float): Clockwise rotation angle

    Returns:
        MatLike: Image centre rotated by angle. 
    """
    h, w, _ = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def spin(img: MatLike):
    """Visualise the image rotating.

    Args:
        img (MatLike): Image to rotate
    """
    for i in range(0, 1000, 10):
        rot = rotate(img, i)
        cv2.imshow("Rotated", rot)
        cv2.waitKey(50)
    cv2.destroyAllWindows()

def spin_func(
    img: MatLike,
    func: Callable[[MatLike], MatLike],
    iter: int =360,
    deg: int =1,
    time: int =50,
    outfolder: Optional[str]=None,
    index: int =0
) -> None:
    """Rotate, and apply a function to an image. 

    Args:
        img (MatLike): Image
        func (Callable[[MatLike], MatLike]): Modifying function to apply to image. 
        iter (int, optional): Number of times to apply function and rotation. Defaults to 360.
        deg (int, optional): Degrees to rotate per iteration. Defaults to 1.
        time (int, optional): Wait peroid between application (ms). Defaults to 50.
        outfolder (Optional[str], optional): Folder to store intermediate images. Defaults to None.
        index (int, optional): For enumerating image paths. Defaults to 0.
    """
    if outfolder is not None:
        write = True
        print(f"Writing to {outfolder}")
        os.makedirs(outfolder, exist_ok=True)
    else:
        write = False
    for i in range(0, iter, deg):
        rot = rotate(img, i)
        rot = func(rot)
        cv2.imshow("Rotated", rot)
        if write:
            cv2.imwrite(f"{outfolder}/{i + index}.jpg", rot)
        key = cv2.waitKey(time)
        if key == 27:
            break

if __name__ == "__main__":
    print('Rotations')
