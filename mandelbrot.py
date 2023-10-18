from PIL import Image, ImageDraw

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

image = Image.new('RGB', (2000, 2000))
x_width, y_height = image.size

for x in range(x_width):
    for y in range(y_height):
        iterations = iterations_until_divergence(pixel_to_complex(x, y))
        color = 255 - int(iterations / MAX_ITERATIONS * 255)
        image.putpixel((x, y), (color, color, color))

rectangle_width = 100
rectangle_height = 100

draw_image = ImageDraw.Draw(image, 'L')

image.save(f'./output.png')
