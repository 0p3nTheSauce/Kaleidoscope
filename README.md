# Kaleidoscope

Create mirrored images and kaleidoscope videos with multiple planes of symmetry.

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
python mirror.py multi_mirror input.jpg -pn 0 -ot kaleidoscope_0.jpg

# Create a kaleidoscope video (360° rotation)
python mirror.py spin_mirror input.jpg -pn 0 -ov kaleidoscope.mp4
```

## Usage

### Commands Overview

#### `mirror` - Mirror entire image across a plane

Mirror the complete image across a single plane of symmetry.

```bash
python mirror.py mirror INPUT_IMAGE PLANE [OPTIONS]
```

**Planes:**
- `h` - Horizontal
- `v` - Vertical  
- `p` - Positive diagonal (/)
- `n` - Negative diagonal (\)

**Example:**
```bash
python mirror.py mirror photo.jpg v -ot mirrored.jpg
```

---

#### `half_mirror` - Reflect one side onto the other

Reflect one half of the image onto the other side to create a single plane of symmetry.

```bash
python mirror.py half_mirror INPUT_IMAGE SIDE+PLANE [OPTIONS]
```

**Side + Plane combinations:**
- **Sides:** `t` (top), `b` (bottom), `l` (left), `r` (right)
- **Planes:** `h` (horizontal), `v` (vertical), `p` (+diagonal), `n` (-diagonal)

**Valid combinations:** `th`, `bh`, `lv`, `rv`, `tp`, `bp`, `tn`, `bn`

**Example:**
```bash
python mirror.py half_mirror photo.jpg lv -ot symmetric.jpg
```

---

#### `multi_mirror` - Apply 4 planes of symmetry

Create one of 8 possible kaleidoscope patterns by applying 4 planes of symmetry.

```bash
python mirror.py multi_mirror INPUT_IMAGE (-pn PERMUTATION | -cb PLANES...) [OPTIONS]
```

**Permutations (0-7):** Different combinations of 4 symmetry planes

**Custom combinations:** Specify your own sequence of planes (e.g., `-cb lv th tp tn`)

**Example:**
```bash
python mirror.py multi_mirror photo.jpg -pn 0 -ot kaleidoscope.jpg
python mirror.py multi_mirror photo.jpg -cb lv th tp bn -ot custom.jpg
```

---

#### `spin_mirror` - Create rotating kaleidoscope videos

Rotate the image while applying multi-mirror transformations to create mesmerizing kaleidoscope animations.

```bash
python mirror.py spin_mirror INPUT_IMAGE (-pn PERMUTATION | -cb PLANES...) [OPTIONS]
```

**Key options:**
- `-ov OUTPUT.mp4` - Save as video
- `-od OUTPUT_DIR/` - Save individual frames
- `-it N` - Number of iterations (default: 360)
- `-dg N` - Degrees per iteration (default: 1)
- `-fr FPS` - Frame rate (default: 30.0)

**Example:**
```bash
python mirror.py spin_mirror photo.jpg -pn 0 -ov kaleidoscope.mp4 -fr 30
python mirror.py spin_mirror photo.jpg -pn 0 -it 180 -dg 2 -ov fast.mp4
```

---

### Common Options

**Image preprocessing:**
- `-sq, --square` - Crop to center square before operations
- `-ns WIDTH HEIGHT` - Resize to specific dimensions (e.g., `-ns 1920 1080`)
- `-fx FACTOR` - Scale width by factor
- `-fy FACTOR` - Scale height by factor

**Display:**
- `-nd, --no_disp` - Don't display output
- `-do, --disp_original` - Display original image
- `-dv, --disp_verbose` - Show intermediate steps (multi_mirror only)

**Output:**
- `-ot PATH` - Save output image
- `-od DIR` - Save frames to directory (spin_mirror only)
- `-ov PATH` - Save output video (spin_mirror only)

---

### Full Command Reference

<details>
<summary><b>Click to see complete help output</b></summary>

#### Main help
```
usage: mirror.py [-h] {mirror,half_mirror,multi_mirror,spin_mirror} ...

positional arguments:
  {mirror,half_mirror,multi_mirror,spin_mirror}
                        Available commands
    mirror              Mirror a whole image
    half_mirror         Reflect one half of the image onto the other
    multi_mirror        Create one of the 8 possible images produced by applying 4 planes of symmetry.
    spin_mirror         Rotate image while applying multi-mirror (creates kaleidoscope)

optional arguments:
  -h, --help            show this help message and exit
```

#### mirror command
```
usage: mirror.py mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-fx FACTOR_X] [-fy FACTOR_Y] 
                        [-nd] [-do] [-ot OUT_IMG] in_img {v,h,p,n}

positional arguments:
  in_img                Path to input image
  plane                 Mirror about this plane of symmetry: Planes: h/v/p/n 
                        (horizontal/vertical/+diagonal/-diagonal)

optional arguments:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -ot OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
```

#### half_mirror command
```
usage: mirror.py half_mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-fx FACTOR_X] [-fy FACTOR_Y]
                             [-nd] [-do] [-ot OUT_IMG] in_img SIDE+PLANE

positional arguments:
  in_img                Path to input image
  SIDE+PLANE            Side to reflect + plane of symmetry (e.g. th). 
                        Sides: t/b/l/r (top/bottom/left/right). 
                        Planes: h/v/p/n (horizontal/vertical/+diagonal/-diagonal).

optional arguments:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -ot OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
```

#### multi_mirror command
```
usage: mirror.py multi_mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-fx FACTOR_X] [-fy FACTOR_Y]
                              [-nd] [-do] [-ot OUT_IMG] (-pn {0,1,2,3,4,5,6,7} | -cb PLANE [PLANE ...])
                              [-dv] in_img

positional arguments:
  in_img                Path to input image

optional arguments:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
  -fx FACTOR_X, --factor_x FACTOR_X
                        Factor to multiply width by (resize)
  -fy FACTOR_Y, --factor_y FACTOR_Y
                        Factor to multiply height by (resize)
  -nd, --no_disp        Do not view output
  -do, --disp_original  View original image
  -ot OUT_IMG, --out_img OUT_IMG
                        Output path of image. Otherwise don't save
  -pn {0,1,2,3,4,5,6,7}, --perm_num {0,1,2,3,4,5,6,7}
                        Permutation of operations [0-7].
  -cb PLANE [PLANE ...], --comb PLANE [PLANE ...]
                        Custom combination of planes (e.g., lv th tp tn)
  -dv, --disp_verbose   Display intermediary steps
```

#### spin_mirror command
```
usage: mirror.py spin_mirror [-h] [-sq] [-ns WIDTH HEIGHT] [-fx FACTOR_X] [-fy FACTOR_Y]
                             [-nd] [-do] (-pn {0,1,2,3,4,5,6,7} | -cb PLANE [PLANE ...])
                             [-it ITERATIONS] [-dg DEGREES] [-wt WAIT] [-ix INDEX]
                             [-od OUT_DIR] [-ov OUT_VID] [-fr FRAME_RATE] [-vc VIDEO_CODE]
                             [-nr] in_img

positional arguments:
  in_img                Path to input image

optional arguments:
  -h, --help            show this help message and exit
  -sq, --square         Crop to centre square before other operations
  -ns WIDTH HEIGHT, --new_size WIDTH HEIGHT
                        New size as width height (e.g., -ns 1920 1080)
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
  -it ITERATIONS, --iterations ITERATIONS
                        Number of times to apply function and rotation. Defaults to 360.
  -dg DEGREES, --degrees DEGREES
                        Degrees to rotate per iteration. Defaults to 1.
  -wt WAIT, --wait WAIT
                        Wait period between application (ms). Defaults to 1.
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

### Simple Mirror
```bash
# Vertical mirror (left-right flip)
python mirror.py mirror landscape.jpg v -ot mirrored_v.jpg

# Horizontal mirror (top-bottom flip)
python mirror.py mirror portrait.jpg h -ot mirrored_h.jpg
```

### Half Mirror (Create Symmetry)
```bash
# Reflect left side onto right (vertical symmetry)
python mirror.py half_mirror face.jpg lv -ot symmetric_face.jpg

# Reflect top onto bottom (horizontal symmetry)
python mirror.py half_mirror building.jpg th -ot symmetric_building.jpg
```

### Multi-Mirror Kaleidoscope
```bash
# Create kaleidoscope pattern with 4 planes of symmetry
python mirror.py multi_mirror flower.jpg -pn 0 -sq -ot kaleidoscope.jpg

# Custom plane combination
python mirror.py multi_mirror texture.jpg -cb lv th tp tn -ot custom_pattern.jpg
```

### Rotating Kaleidoscope Video
```bash
# Standard 360° rotation at 30 fps
python mirror.py spin_mirror sunset.jpg -pn 0 -sq -ov sunset_kaleidoscope.mp4

# Fast rotation: 180 iterations, 2° per step
python mirror.py spin_mirror abstract.jpg -pn 3 -it 180 -dg 2 -fr 60 -ov fast_spin.mp4

# Save individual frames to directory
python mirror.py spin_mirror mandala.jpg -pn 0 -od ./frames/ -sq
```

### Image Preprocessing
```bash
# Resize to HD and create kaleidoscope
python mirror.py multi_mirror photo.jpg -ns 1920 1080 -sq -pn 0 -ot hd_kaleidoscope.jpg

# Scale by factor and mirror
python mirror.py mirror large_image.jpg h -fx 0.5 -fy 0.5 -ot smaller_mirror.jpg
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