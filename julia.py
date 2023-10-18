from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time


CONVERGENCE_THRESHOLD = 2
MAX_ITERATIONS = 100
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2


@njit(parallel=True)
def njit_mandelbrot_array_fast(x_width, y_height, c):
    iterations = np.empty((y_height, x_width))
    for x in prange(x_width):
        for y in prange(y_height):
            z = complex(X_MIN + (x / x_width) * (X_MAX - X_MIN), Y_MIN + (y / y_height) * (Y_MAX - Y_MIN))
            i = 0
            for _ in range(MAX_ITERATIONS):
                z = z**2 + c
                if abs(z) > CONVERGENCE_THRESHOLD:
                    break
                i+=1
            iterations[x, y] = i
    
    iterations = iterations.transpose()
    return 255 - (iterations / MAX_ITERATIONS * 255)

if __name__ == '__main__':
    start_time = time.time()

    image = Image.new('L', (2000, 2000))
    x_width, y_height = image.size

    img_arr = njit_mandelbrot_array_fast(x_width, y_height, c = complex(-0.8, 0.156))
    image.putdata(img_arr.flatten().tolist())
    image.save(f'./output_np.png')

    print(f'Runtime: {time.time() - start_time}')

    plt.style.use('default')
    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')
    plt.show()

