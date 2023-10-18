from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time


CONVERGENCE_THRESHOLD = 2
MAX_ITERATIONS = 100
X_MIN, X_MAX = -2.5, 1.5
Y_MIN, Y_MAX = -2, 2


def pixel_to_complex(x, y):
    return complex(X_MIN + (x / x_width) * (X_MAX - X_MIN), Y_MIN + (y / y_height) * (Y_MAX - Y_MIN))


def iterations_until_divergence(c):
    z = 0
    for iteration in range(MAX_ITERATIONS):
        z = z**2 + c
        if abs(z) > CONVERGENCE_THRESHOLD:
            return iteration
    return MAX_ITERATIONS


def ndenumberate_mandelbrot_array(x_width, y_height):
    iterations = np.empty((y_height, x_width))

    for ((i, j), _)in np.ndenumerate(iterations):
        iterations[i, j] = iterations_until_divergence(pixel_to_complex(i, j))

    return 255 - (iterations.transpose() / MAX_ITERATIONS * 255)


def mandelbrot_array(x_width, y_height):
    iterations = np.fromfunction(lambda x, y: np.vectorize(iterations_until_divergence)(np.vectorize(pixel_to_complex)(x, y)), (y_height, x_width), dtype=int).transpose()
    return 255 - iterations / MAX_ITERATIONS * 255


@njit
def njit_mandelbrot_array(x_width, y_height):
    iterations = np.empty((y_height, x_width))

    for ((x, y), _) in np.ndenumerate(iterations):
        c = complex(X_MIN + (x / x_width) * (X_MAX - X_MIN), Y_MIN + (y / y_height) * (Y_MAX - Y_MIN))
        z = 0
        i = 0
        for _ in range(MAX_ITERATIONS):
            z = z**2 + c
            if abs(z) > CONVERGENCE_THRESHOLD:
                break
            i+=1
        iterations[x, y] = i
    
    iterations = iterations.transpose()

    return 255 - (iterations / MAX_ITERATIONS * 255)

image = Image.new('L', (20000, 20000))
x_width, y_height = image.size

# start_time = time.time()
# img_arr = ndenumberate_mandelbrot_array(x_width, y_height)
# print(f'ndenumberate_mandelbrot_array: {time.time() - start_time}')

# start_time = time.time()
# img_arr = mandelbrot_array(x_width, y_height)
# print(f'mandelbrot_array: {time.time() - start_time}')

start_time = time.time()
img_arr = njit_mandelbrot_array(x_width, y_height)
print(f'njit_mandelbrot_array: {time.time() - start_time}')

image.putdata(img_arr.flatten().tolist())

image.save(f'./output_np.png')

plt.style.use('default')
plt.imshow(img_arr, cmap='gray')
plt.axis('off')
plt.show()
