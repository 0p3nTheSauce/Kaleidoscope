import cv2
import numpy as np
import time
from numba import njit

CV_FLIP_HORIZ = 1

# Placeholder for remove_diag_n (replace with your actual function)
@njit(cache=True)
def remove_diag_n(img):
    # Your actual implementation here
    pass

def remove_diag_p(img):
    cv2.flip(img, CV_FLIP_HORIZ, dst=img)
    remove_diag_n(img)
    return cv2.flip(img, CV_FLIP_HORIZ, dst=img)

@njit(cache=True)
def remove_diag_p_n(img):
    img[:] = img[:, ::-1]
    remove_diag_n(img)
    img[:] = img[:, ::-1]
    return img

# Create test image
img_size = (1000, 1000, 3)  # Adjust size as needed
test_img = np.random.randint(0, 256, img_size, dtype=np.uint8)

# Warm up numba JIT
test_copy = test_img.copy()
remove_diag_p_n(test_copy)

# Benchmark cv2.flip version
n_iterations = 1000
times_cv2 = []

for _ in range(n_iterations):
    img_copy = test_img.copy()
    start = time.perf_counter()
    remove_diag_p(img_copy)
    end = time.perf_counter()
    times_cv2.append(end - start)

# Benchmark NumPy version
times_numpy = []

for _ in range(n_iterations):
    img_copy = test_img.copy()
    start = time.perf_counter()
    remove_diag_p_n(img_copy)
    end = time.perf_counter()
    times_numpy.append(end - start)

# Results
avg_cv2 = np.mean(times_cv2) * 1000  # Convert to ms
avg_numpy = np.mean(times_numpy) * 1000
std_cv2 = np.std(times_cv2) * 1000
std_numpy = np.std(times_numpy) * 1000

print(f"Image size: {img_size}")
print(f"Iterations: {n_iterations}\n")
print(f"cv2.flip version:")
print(f"  Mean: {avg_cv2:.4f} ms")
print(f"  Std:  {std_cv2:.4f} ms\n")
print(f"NumPy version:")
print(f"  Mean: {avg_numpy:.4f} ms")
print(f"  Std:  {std_numpy:.4f} ms\n")
print(f"Speedup: {avg_cv2/avg_numpy:.2f}x")