from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import random
from time import sleep


CONVERGENCE_THRESHOLD = 2
MAX_ITERATIONS = 100
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2


@njit(parallel=True)
def juilia_set(x_width, y_height, c):
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
    for _ in range(100):
        start_time = time.time()

        image = Image.new('L', (2000, 2000))n
        x_width, y_height = image.size

        constant = complex(round(random.random() * random.choice([1, -1]), 3),round(random.random() * random.choice([1, -1]), 3))

        img_arr = juilia_set(x_width, y_height, c = constant)
        image.putdata(img_arr.flatten().tolist())

        print(f'Runtime: {time.time() - start_time}')

        plt.style.use('default')
        plt.imshow(img_arr, cmap='gray')
        plt.axis('off')
        plt.show()
        cool = input("Cool? ")
        if cool == 'y':
            image.save(f'./julia/julia_cool_{constant.real}_{constant.imag}.png')
        else:
            image.save(f'./julia/julia_{constant.real}_{constant.imag}.png')

