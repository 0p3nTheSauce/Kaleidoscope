# Kaleidoscope

Create mirrored images and kaleidoscope videos with multiple planes of symmetry. Liew view or save outputs. Implemented with openCV, use Esc or Shift to close windows.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Commands Overview](#commands-overview)
  - [Full Command Reference](#full-command-reference)
- [Examples](#examples)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
git clone https://github.com/0p3nTheSauce/Kaleidoscope.git
cd Kaleidoscope

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Reflect images across 4-planes of symmetry
python mirror.py multi_mirror input.jpg -oi kaleidoscope_0.jpg

# Create a kaleidoscope video (360° rotation)
python mirror.py spin_mirror input.jpg -ov kaleidoscope.mp4
```

## Usage

### Commands Overview

#### `view` - View the immage with no mirroring

Useful for previewing the image with resizing and square cropping

```bash
python mirror.py view INPUT_IMAGE [OPTIONS]
```

#### `mirror` - Mirror entire image across a plane

Mirror the complete image across a single plane of symmetry.

```bash
python mirror.py mirror INPUT_IMAGE PLANE [OPTIONS]
```

**Planes:**
- `h` - Horizontal
- `v` - Vertical  
- `p` - Positive diagonal 
- `n` - Negative diagonal 

**Example:**
```bash
python mirror.py mirror input.jpg v -oi mirrored.jpg
```

---

#### `multi_mirror` - Apply multiple planes of symmetry

Four planes of symmetry have been implemented, if all are applied, there are 8 possible resulting images (for a given crop). Each of these images can be selected with the [-pn] flag. Otherwise custom plane combinations can be used with the [-cb] flag.

```bash
python mirror.py multi_mirror INPUT_IMAGE (-pn PERMUTATION | -cb PLANES...) [OPTIONS]
```

**Side + Plane combinations:**
- **Sides:** `t` (top), `b` (bottom), `l` (left), `r` (right)
- **Planes:** `h` (horizontal), `v` (vertical), `p` (+diagonal), `n` (-diagonal)

**Valid combinations:** `th`, `bh`, `lv`, `rv`, `tp`, `bp`, `tn`, `bn`


**Permutations (0-7):** Combinations of 4 planes of symmetry which produce unique images (e.g. lv th tp tn) .

**Custom combinations:** Apply variable number or combination of mirrors (e.g. `-cb lv th`)

**Example:**
```bash
python mirror.py multi_mirror photo.jpg -pn 0 -oi kaleidoscope.jpg
python mirror.py multi_mirror photo.jpg -cb lv th -oi custom.jpg
```

---

#### `spin_mirror` - Create rotating kaleidoscope videos

Rotate the image while applying multi-mirror transformations to create kaleidoscopic animations. By default, it produces one full rotation of the image (360 degrees), applying 8 planes of symmetry every 1 degree. Press Esc to terminated live view.

```bash
python mirror.py spin_mirror INPUT_IMAGE [OPTIONS]
```

**Key options:**
- `-pn PERMUTATION_NUM` - Same as multi_mirror
- `-cb PLANES - Same as multi_mirror
- `-ov OUTPUT.mp4` - Save as video
- `-od OUTPUT_DIR/` - Save individual frames
- `-it N` - Number of iterations (default: 360)
- `-dg N` - Degrees per iteration (default: 1)
- `-wt N` - Wait time (ms) between viewing individual frames during live output (cv2.waitKey style)
- `-fr FPS` - Frame rate of output video (default: 30.0)

**Example:**
```bash
python mirror.py spin_mirror photo.jpg -ov kaleidoscope.mp4 
python mirror.py spin_mirror photo.jpg -it 180 -dg 2 -ov -fr 60 fast.mp4
```

---

### Common Options

**Image preprocessing:**
- `-sq, --square` - Crop to center square before operations (important when mirroring about diagonals)
- `-as, --auto_size` - Automatically scale window based on screen size
- `-ns WIDTH HEIGHT` - Resize to specific dimensions (e.g., `-ns 1920 1080`)
- `-fx FACTOR` - Scale width by factor
- `-fy FACTOR` - Scale height by factor

**Display:**
- `-nd, --no_disp` - Don't display live view
- `-do, --disp_original` - Display original image
- `-dv, --disp_verbose` - Show intermediate steps (multi_mirror only)

**Output:**
- `-oi PATH` - Save output image
- `-od DIR` - Save frames to directory (spin_mirror only)
- `-ov PATH` - Save output video (spin_mirror only)

---

### Full Command Reference

<details>
<summary><b>Click to see complete help output</b></summary>

#### Main help
```
usage: mirror.py [-h] {view,mirror,multi_mirror,spin_mirror} ...

positional arguments:
  {view,mirror,multi_mirror,spin_mirror}
                        Available commands
    view                Don't apply any mirroring
    mirror              Mirror a whole image
    multi_mirror        Create images with up to 4 planes of symmetry
    spin_mirror         Rotate image while applying multi_mirror (creates kaleidoscope)

options:
  -h, --help            show this help message and exit
```

#### view command
```
usage: mirror.py view [-h] [-sq] [-ns WIDTH HEIGHT] [-as] [-fx FACTOR_X] [-fy FACTOR_Y] [-nd] [-do] [-oi OUT_IMG] in_img

positional arguments:
  in_img                Path to input image

options:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -as, --auto_size
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -oi OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
```

#### mirror command
```
usage: mirror.py mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-as] [-fx FACTOR_X] [-fy FACTOR_Y] [-nd] [-do] [-oi OUT_IMG] in_img {v,h,p,n}

positional arguments:
  in_img                Path to input image
  {v,h,p,n}             Mirror about this plane of symmetry: Planes: h/v/p/n (horizontal/vertical/+diagonal/-diagonal)

options:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -as, --auto_size
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -oi OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
```

#### multi_mirror command
```
usage: mirror.py multi_mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-as] [-fx FACTOR_X] [-fy FACTOR_Y] [-nd] [-do] [-oi OUT_IMG] (-pn {0,1,2,3,4,5,6,7} | -cb SIDE+PLANE [SIDE+PLANE ...]) [-dv] [-f] in_img

positional arguments:
  in_img                Path to input image

options:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -as, --auto_size
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -oi OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
  -pn {0,1,2,3,4,5,6,7}, --perm_num {0,1,2,3,4,5,6,7}
                        Permutation of operations [0-7].
  -cb SIDE+PLANE [SIDE+PLANE ...], --comb SIDE+PLANE [SIDE+PLANE ...]
                        Side to reflect + plane of symmetry (e.g. th). Sides: t/b/l/r (top/bottom/left/right). Planes: h/v/p/n (horizontal/vertical/+diagonal/-diagonal).
  -dv, --disp_verbose   Display intermediary steps
  -f, --force           Force multi_mirror to use rectangular images when using diagonals.
```

#### spin_mirror command
```
usage: mirror.py spin_mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-as] [-fx FACTOR_X] [-fy FACTOR_Y] [-nd] [-do] [-pn {0,1,2,3,4,5,6,7} | -cb PLANE [PLANE ...]] [-f] [-it ITERATIONS] [-dg DEGREES] [-wt WAIT]
                             [-ix INDEX] [-od OUT_DIR] [-ov OUT_VID] [-fr FRAME_RATE] [-vc VIDEO_CODE] [-nr]
                             in_img

positional arguments:
  in_img                Path to input image

options:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -as, --auto_size
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -pn {0,1,2,3,4,5,6,7}, --perm_num {0,1,2,3,4,5,6,7}
                        Permutation of operations [0-7].
  -cb PLANE [PLANE ...], --comb PLANE [PLANE ...]
                        Custom combination of planes (e.g., lv th tp tn)
  -f, --force           Force spin_mirror to use rectangular images.
  -it ITERATIONS, --iterations ITERATIONS
                        Number of times to apply function and rotation. Defaults to 360.
  -dg DEGREES, --degrees DEGREES
                        Degrees to rotate per iteration. Defaults to 1.
  -wt WAIT, --wait WAIT
                        Wait peroid between application (ms). Defaults to 1.
  -ix INDEX, --index INDEX
                        For enumerating image paths. Defaults to 0.
  -od OUT_DIR, --out_dir OUT_DIR
                        Output path. Save intermediary images to a directory
  -ov OUT_VID, --out_vid OUT_VID
                        Output path. Create a video from the images
  -fr FRAME_RATE, --frame_rate FRAME_RATE
                        Frame rate of output video
  -vc VIDEO_CODE, --video_code VIDEO_CODE
                        Video codec for output video
  -nr, --no_recode      Don't recode the video with FFMPEG
```

</details>

## Examples

### View

```bash
#resize and crop square
python mirror.py view examples/Fern.png -fx 0.15 -fy 0.15 -sq -oi examples/Fern_as_sq.png
```

![Auto-sized and square cropped Fern.png](./examples/Fern_as_sq.png)

### Simple Mirror
```bash
# Vertical mirror (left-right flip)
python mirror.py mirror ./examples/Fern_as_sq.png v -oi ./examples/Fern_v.png

```

![Horizontally flipped Fern.png](./examples/Fern_h.png)

### Multi-Mirror
```bash
# Create kaleidoscope pattern with 4 planes of symmetry
python mirror.py multi_mirror ./examples/Fern_as_sq.png -pn 0 -oi ./examples/Fern_0.png
```

![Four plane symmetry Fern.png](./examples/Fern_0.png)

### Rotating Multi-Mirror
```bash
# Standard 360° rotation at 30 fps
python mirror.py spin_mirror sunset.jpg -sq -ov sunset_kaleidoscope.mp4
```

## Features

- **Multiple mirror modes:**
  - Simple mirroring across horizontal, vertical, and diagonal planes
  - Half-mirror reflection to create single-plane symmetry
  - Multi-mirror for complex kaleidoscope patterns with 4 planes of symmetry
  
- **Kaleidoscope video generation:**
  - Rotating animations with customizable rotation speed
  - Adjustable frame rate and video codec
  - Option to save individual frames
  
- **Flexible image processing:**
  - Automatic center square cropping
  - Resize by dimensions or scaling factors
  - Support for various image formats
  
- **Customization:**
  - 8 built-in symmetry permutations
  - Custom plane combinations
  - Verbose display of intermediate steps

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Numba
- FFmpeg (optional, for video re-encoding)

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share your creations

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for creating beautiful symmetrical art**