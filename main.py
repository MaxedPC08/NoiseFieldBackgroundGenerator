import time
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import imageio
from skimage.transform import resize
import threading
import os
import argparse

def generate_noise(width, height, scale=10):
    def simplex(x, y, p, grad3):
        s = (x + y) * 0.5 * (np.sqrt(3.0) - 1.0)
        i = int(x + s)
        j = int(y + s)
        t = (i + j) * (3.0 - np.sqrt(3.0)) / 6.0
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + (3.0 - np.sqrt(3.0)) / 6.0
        y1 = y0 - j1 + (3.0 - np.sqrt(3.0)) / 6.0
        x2 = x0 - 1.0 + (3.0 - np.sqrt(3.0)) / 3.0
        y2 = y0 - 1.0 + (3.0 - np.sqrt(3.0)) / 3.0

        ii = i & 255
        jj = j & 255
        gi0 = p[ii + p[jj]] % 12
        gi1 = p[ii + i1 + p[jj + j1]] % 12
        gi2 = p[ii + 1 + p[jj + 1]] % 12

        t0 = 0.5 - x0 * x0 - y0 * y0
        n0 = 0.0 if t0 < 0 else (t0 * t0) * (t0 * t0) * (grad3[gi0, 0] * x0 + grad3[gi0, 1] * y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        n1 = 0.0 if t1 < 0 else (t1 * t1) * (t1 * t1) * (grad3[gi1, 0] * x1 + grad3[gi1, 1] * y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        n2 = 0.0 if t2 < 0 else (t2 * t2) * (t2 * t2) * (grad3[gi2, 0] * x2 + grad3[gi2, 1] * y2)

        return 70.0 * (n0 + n1 + n2)

    grad3 = np.array([[1,1],[-1,1],[1,-1],[-1,-1],
                      [1,0],[-1,0],[1,0],[-1,0],
                      [0,1],[0,-1],[0,1],[0,-1]], dtype=np.float32)

    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    p = np.concatenate((p, p))

    noise = np.zeros((width, height), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            noise[i, j] = simplex(i / scale, j / scale, p, grad3)

    noise = (noise - np.min(noise)) / np.ptp(noise)
    return noise

def generate_perlin_noise(width, height, scale=10):
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(t, a, b):
        return a + t * (b - a)

    def grad(hash, x, y):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else 0)
        return ((u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v))

    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    p = np.concatenate((p, p))

    noise = np.zeros((width, height), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            x = i / scale
            y = j / scale

            X = int(np.floor(x)) & 255
            Y = int(np.floor(y)) & 255

            x -= np.floor(x)
            y -= np.floor(y)

            u = fade(x)
            v = fade(y)

            n00 = grad(p[p[X] + Y], x, y)
            n01 = grad(p[p[X] + Y + 1], x, y - 1)
            n10 = grad(p[p[X + 1] + Y], x - 1, y)
            n11 = grad(p[p[X + 1] + Y + 1], x - 1, y - 1)

            noise[i, j] = lerp(v, lerp(u, n00, n10), lerp(u, n01, n11))

    noise = (noise - np.min(noise)) / np.ptp(noise)
    return noise

def optimized_perlin_noise(x, y, scale, upscale = 20):
    image = generate_perlin_noise(x//upscale+1, y//upscale+1, scale)
    image = resize(image, (x, y), order=1, mode='reflect', anti_aliasing=True)
    return image

def optimized_noise(x, y, scale, upscale = 20):
    image = generate_noise(x//upscale+1, y//upscale+1, scale)
    image = gaussian_filter(np.kron(image, np.ones((upscale, upscale))), sigma=2*scale)[:x, :y]
    return image


def draw_perlin_gradient(width, height, scale=6, upscale=20, color_start=(0, 0, 0), color_end=(255, 255, 255)):
    # Make a noise field? TODO: how does this actually work
    noise = optimized_perlin_noise(width, height, scale, upscale)

    # grads lmao
    color_start = np.array(color_start, dtype=np.float32)
    color_end = np.array(color_end, dtype=np.float32)
    gradient = (color_start + noise[..., np.newaxis] * (color_end - color_start)).astype(np.uint8)

    return gradient

def generate_analogous_colors(hue=None):
    def hsv_to_rgb(h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h * 6.0)  # Assume h is in [0, 1]
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

    # make a color
    if hue is None:
        hue = random.random()
        # Saturation and value are set to 1 for vivid colors, but idk maybe try something less egregious at somepoint. Pastel mode would be pretty sick
        color1 = hsv_to_rgb(hue, 1, 1)
        # Complementary color is 180 degrees apparently
        color2 = hsv_to_rgb((hue + 1/4) % 1.0, 1, 1)
    else:
        offset = random.random() * 0.25 - 0.125
        color1 = hsv_to_rgb(hue + offset, 1, 1)
        color2 = hsv_to_rgb((hue + 1/4 + offset) % 1.0, 1, 1)

    # 8biterizer
    color1 = tuple(int(c * 255) for c in color1)
    color2 = tuple(int(c * 255) for c in color2)

    return color1, color2

def draw_perlin_holes(width, height, scale=6, upscale=20, threshold=0.5, num_colors=3, max_grads=5, hue=np.random.random()):
    noise = optimized_noise(width, height, scale, upscale)
    def generate_color_thread(colors, index, hue):
        colors[index] = draw_perlin_gradient(width, height, scale, upscale, *generate_analogous_colors(hue))

    colors = [None] * min(max_grads, num_colors)
    threads = []
    for i in range(min(max_grads, num_colors)):
        thread = threading.Thread(target=generate_color_thread, args=(colors, i, hue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    # colors = [draw_perlin_gradient(width, height, scale, upscale, *generate_analogous_colors(hue)) for _ in range(min(max_grads, num_colors))]

    masks = np.zeros((width, height), dtype=int)
    for i in range(num_colors):
        masks[noise >= threshold + i * (1 - threshold) / num_colors] = i

    masks-=1
    image = np.zeros((width, height, 3), dtype=np.uint8)
    for i in range(num_colors):
        image[masks == i] = colors[i%max_grads][masks == i]

    noise -= threshold
    noise %= (1 - threshold) / num_colors
    noise = (noise - noise.min()) / np.ptp(noise)
    noise = 1 - noise
    noise **= (0.25)
    noise = noise[..., np.newaxis]

    noise = (noise + gaussian_filter(noise, sigma=(3, 3, 0)))/2
    image = np.multiply(image, noise).astype(np.uint8)


    return image
    

def main(w, h, hue, filename=None):
    
    width, height = w, h
    scale = 15
    upscale = 40
    gradient_image = draw_perlin_holes(width, height, scale, upscale, threshold=0.2, num_colors=10, max_grads=5, hue=hue)
    # I don't love just overriding the pictures but its fine casue I have two different monitor resolutions
    if not filename:
        if not os.path.exists('.images'):
            os.makedirs('.images')
        imageio.imwrite('.images/perlin_gradient_'+str(w)+'_'+str(h)+'.png', gradient_image)
    else:
        imageio.imwrite(filename, gradient_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Perlin noise images.")
    parser.add_argument("--save", action="store_true", help="Save the generated image to disk.")
    parser.add_argument("--width", type=int, default=1440, help="Width of the image (default: 1440)")
    parser.add_argument("--height", type=int, default=2560, help="Height of the image (default: 2560)")
    parser.add_argument("--hue", type=float, default=None, help="Hue for colors (random if not specified)")
    parser.add_argument("--fname", type=str, default=None, help="File Name to save the generated image under")
    args = parser.parse_args()

    if args.save:
        hue = args.hue if args.hue is not None else np.random.random()
        main(args.width, args.height, hue, args.fname)
    else:
        start = time.time()
        hue = args.hue if args.hue is not None else np.random.random()
        thread1 = threading.Thread(target=main, args=(args.width, args.height, hue))
        thread2 = threading.Thread(target=main, args=(1080, 1920, hue))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()
        print('Time taken: %.4f seconds' % (time.time() - start))
