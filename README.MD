PerlinDrawer
A Python script for generating beautiful Perlin noise-based images with gradients and "holes" effects. It uses Simplex and Perlin noise algorithms to create visually appealing patterns, optimized for performance with upscaling and Gaussian filtering.

##Features
- Generate Perlin and Simplex noise fields
- Create gradient images from one color to another
- Produce "holes" effects with multiple colors and thresholds
- Multithreaded generation for faster processing
- Command-line interface for easy usage

##Dependencies
- numpy
- scipy
- imageio
- scikit-image (skimage)
Install them via pip:
```pip install numpy scipy imageio scikit-image```
Usage
Run the script from the command line:
```python main.py```

### Options
- `--save`: Save the generated image to disk. If not specified, generates images for two resolutions (1440x2560 and 1080x1920) using multithreading and prints the time taken.
- `--width`: Width of the image (default: 1440). Used for the first resolution when --save is specified, or for the custom width in default mode.
- `--height`: Height of the image (default: 2560). Used for the first resolution when --save is specified, or for the custom height in default mode.
- `--hue`: Hue for colors (random if not specified). A float value between 0 and 1.
- `--fname`: File name to save the generated image under (only used with --save). If not specified, saves to `.images/perlin_gradient_<width>_<height>.png`.

Examples are in the images directory
