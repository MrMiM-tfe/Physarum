import cv2
import numpy as np
import math
import os
import threading


WIDTH = HEIGHT = 600
AGENT_SIZE = 1
NUM_AGENTS = 1000

AGENTS = not True
IN_CENTER = False

SPEED = 2  # pixel/frame
DIR_ANGLE = 40 # deg

SENSOR_ANGLE = 40 # deg
SENSOR_LENGTH = 20 # pixel

PHEROMONE_MIN = 50 # 0
PHEROMONE_MAX = 160 # 255

EVAPORATION = 0.8 # [0, 255]
# DIFFUSION = 0.5
BLUR_SCALE = 5
NUM_BLUR_THREADS = 6

WOBBLING = 10
WOBBLING_CHANCE = 0 # [0, 100]

INSANITY_PHEROMONE = 230 # [0, 255]
INSANITY_CHANCE = 0 # [0, 100]

SCALE = 1
FRAMERATE = 27

# r: random, c: go to center
FIRST_DIRECTION = "c"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED
        self.pheromone = np.random.randint(PHEROMONE_MIN, PHEROMONE_MAX)
        self.insanity = False

        # set Direction
        fr = FIRST_DIRECTION
        if fr == "c":
            center_x = WIDTH / 2
            center_y = HEIGHT / 2

            to_center_vector = np.array([center_x - self.x, center_y - self.y])
            normalized_vector = to_center_vector / np.linalg.norm(to_center_vector)
            self.direction = normalized_vector * SPEED
        elif fr == "r":
            i = np.random.randint(-self.speed, self.speed)
            j = math.sqrt(self.speed**2 - i**2)
            self.direction = np.array([i, j])

    def move(self, image, image2):
        last_x = int(self.x)
        last_y = int(self.y)

        moving_dir = self.direction / np.linalg.norm(self.direction)
        sensor_dir = moving_dir * SENSOR_LENGTH

        sensor_positions = np.array([
            [self.x, self.y],
            [self.x + sensor_dir[0] * math.cos(math.radians(SENSOR_ANGLE)) - sensor_dir[1] * math.sin(math.radians(SENSOR_ANGLE)),
            self.y + sensor_dir[1] * math.cos(math.radians(SENSOR_ANGLE)) + sensor_dir[0] * math.sin(math.radians(SENSOR_ANGLE))],
            [self.x + sensor_dir[0] * math.cos(math.radians(-SENSOR_ANGLE)) - sensor_dir[1] * math.sin(math.radians(-SENSOR_ANGLE)),
            self.y + sensor_dir[1] * math.cos(math.radians(-SENSOR_ANGLE)) + sensor_dir[0] * math.sin(math.radians(-SENSOR_ANGLE))]
        ], dtype=int)

        try:
            # Extract sensor values using NumPy slicing
            sensor_values = image[sensor_positions[:, 1], sensor_positions[:, 0], 2]

            front_sensor_value, left_sensor_value, right_sensor_value = sensor_values

            if np.random.randint(0, 100) >= INSANITY_CHANCE or max(sensor_values) < INSANITY_PHEROMONE:

                self.insanity = False
                # Adjust direction based on red sensor values (must value)
                if front_sensor_value > left_sensor_value and front_sensor_value > right_sensor_value or (
                        right_sensor_value < 1 and left_sensor_value < 1):
                    pass
                elif left_sensor_value > right_sensor_value:
                    self.direction = np.dot(self.direction, np.array([
                        [math.cos(math.radians(-DIR_ANGLE)), -math.sin(math.radians(-DIR_ANGLE))],
                        [math.sin(math.radians(-DIR_ANGLE)), math.cos(math.radians(-DIR_ANGLE))]
                    ]))
                else:
                    self.direction = np.dot(self.direction, np.array([
                        [math.cos(math.radians(DIR_ANGLE)), -math.sin(math.radians(DIR_ANGLE))],
                        [math.sin(math.radians(DIR_ANGLE)), math.cos(math.radians(DIR_ANGLE))]
                    ]))
            else:
                self.insanity = True
                # Adjust direction based on red sensor values (least value)
                if front_sensor_value < left_sensor_value and front_sensor_value < right_sensor_value or (
                        right_sensor_value < 1 and left_sensor_value < 1):
                    pass
                elif left_sensor_value < right_sensor_value:
                    self.direction = np.dot(self.direction, np.array([
                        [math.cos(math.radians(-DIR_ANGLE)), -math.sin(math.radians(-DIR_ANGLE))],
                        [math.sin(math.radians(-DIR_ANGLE)), math.cos(math.radians(-DIR_ANGLE))]
                    ]))
                else:
                    self.direction = np.dot(self.direction, np.array([
                        [math.cos(math.radians(DIR_ANGLE)), -math.sin(math.radians(DIR_ANGLE))],
                        [math.sin(math.radians(DIR_ANGLE)), math.cos(math.radians(DIR_ANGLE))]
                    ]))
        except IndexError:
            pass

        self.x = int(self.x + self.direction[0])
        self.y = int(self.y + self.direction[1])

        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            self.direction *= -1

            self.x = int(self.x + self.direction[0])
            self.y = int(self.y + self.direction[1])

        # wobbling
        wobbling_direction = np.zeros(2)
        if np.random.randint(0, 100) <= WOBBLING_CHANCE:

            i = np.random.randint(-self.speed, self.speed)
            j = math.sqrt(self.speed**2 - i**2)
            wobbling_direction = np.array([i, j])

            self.x = int(self.x + wobbling_direction[0])
            self.y = int(self.y + wobbling_direction[1])

            if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
                wobbling_direction *= -1

                self.x = int(self.x + wobbling_direction[0])
                self.y = int(self.y + wobbling_direction[1])

        # path_pixels = get_line_pixels(last_x, last_y, self.x, self.y)
        path_pixels = get_line_pixels_with_thickness(last_x, last_y, self.x, self.y, 2)

        pheromone_list = []
        for pixel in path_pixels:
            try:
                x, y = pixel
                pheromone = image[y, x][2] + self.pheromone
                image[y, x][2] = min(255, pheromone)
                pheromone_list.append(image[y, x])
            except IndexError:
                pass

        return (path_pixels, pheromone_list)

def red_filter(image):
    # Extract the red channel from the image
    red_channel = image[:, :, 2]

    # Get coordinates where the red component is greater than 0
    red_pixel_coordinates = np.column_stack(np.where(red_channel > 0))

    return red_pixel_coordinates

def get_line_pixels_with_thickness(x1, y1, x2, y2, thickness):
    pixels = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        # Add the original pixel
        pixels.append((x1, y1))

        # Add pixels in a neighborhood around the original pixel
        for i in range(1, thickness + 1):
            for j in range(1, thickness + 1):
                pixels.append((x1 + i, y1 + j))
                pixels.append((x1 + i, y1 - j))
                pixels.append((x1 - i, y1 + j))
                pixels.append((x1 - i, y1 - j))

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels

def get_line_pixels(x1, y1, x2, y2):
    pixels = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        pixels.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels

def apply_gaussian_blur_h(image, start_row, end_row, result):
    # for i in range(start_row, end_row):
    # result[start_row:end_row, :] = cv2.GaussianBlur(image[start_row:end_row, :], (5, 5), 0)
    for i in range(start_row, end_row):
        result[i, :] = cv2.GaussianBlur(image[i, :], (BLUR_SCALE, BLUR_SCALE), 0)

def gaussian_blur_h(image, num_threads=NUM_BLUR_THREADS):
    rows, cols, _ = image.shape
    result = np.zeros_like(image)

    # Create threads
    threads = []
    rows_per_thread = rows // num_threads
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = start_row + rows_per_thread
        if i == num_threads - 1:  # Ensure the last thread processes any remaining rows
            end_row = rows
        thread = threading.Thread(target=apply_gaussian_blur_h, args=(image, start_row, end_row, result))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return result

def apply_gaussian_blur_v(image, start_row, end_row, result):
    for i in range(start_row, end_row):
        result[:, i] = cv2.GaussianBlur(image[:, i], (BLUR_SCALE, BLUR_SCALE), 0)

def gaussian_blur_v(image, num_threads=NUM_BLUR_THREADS):
    cols, rows, _ = image.shape
    result = np.zeros_like(image)

    # Create threads
    threads = []
    rows_per_thread = rows // num_threads
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = start_row + rows_per_thread
        if i == num_threads - 1:  # Ensure the last thread processes any remaining rows
            end_row = rows
        thread = threading.Thread(target=apply_gaussian_blur_v, args=(image, start_row, end_row, result))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return result

def create_gradient(colors):
    image_size = (256,100)
    gradient_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    num_colors = len(colors)
    color_interval = image_size[0] / (num_colors - 1)

    for i in range(num_colors - 1):
        start_idx = int(i * color_interval)
        end_idx = int((i + 1) * color_interval)

        for channel in range(3):
            gradient_image[:, start_idx:end_idx, channel] = np.linspace(
                colors[i][channel], colors[i + 1][channel], end_idx - start_idx
            )

    return gradient_image

def select_circle(matrix, center, radius):
    rows, cols = matrix.shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    return matrix * mask

if IN_CENTER:
    agents = [Agent(np.random.randint(int(WIDTH / 2) - 100, int(WIDTH / 2) + 100), np.random.randint(int(HEIGHT / 2) - 100, int(HEIGHT / 2) + 100)) for _ in range(NUM_AGENTS)]
else:
    agents = [Agent(np.random.randint(WIDTH), np.random.randint(HEIGHT)) for _ in range(NUM_AGENTS)]

# Initialize image with three channels for BGR
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Create a named window for visualization
# cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (mp4v works well)

base_filename = 'videos/simulation_video'
file_extension = '.mp4'
file_number = 1

while True:
    filename = f"{base_filename}_{file_number}{file_extension}"
    if not os.path.exists(filename):
        break
    file_number += 1

video_out = cv2.VideoWriter(filename, fourcc, FRAMERATE, (WIDTH, HEIGHT))

colors = [(131,58,180), (253,29,29)]

# Create the gradient image
gradient = create_gradient(colors)

running = True
frame = 0
while running:
    frame += 1

    image.fill(0)  # Clear the image

    # EVAPORATION
    red_pixels = red_filter(layer1)

    red_pixels = np.array(red_pixels)
    layer1_red_pixels = layer1[red_pixels[:, 0], red_pixels[:, 1]]

    # Update pheromone values using vectorized operations
    layer1_red_pixels[:, 2] = np.maximum(0, layer1_red_pixels[:, 2] * EVAPORATION)

    layer1[red_pixels[:, 0], red_pixels[:, 1]] = layer1_red_pixels

    for agent in agents:
        path, pheromones = agent.move(layer1, image)

        for coordinate, pheromone in zip(path, pheromones):
            try:
                layer1[coordinate[1], coordinate[0]] = pheromone
            except IndexError:
                pass

        # Draw agent
        if AGENTS:
            image[agent.y, agent.x] = (255,0,0) if agent.insanity else (255, 255, 255)

    layer1 = gaussian_blur_h(layer1)
    layer1 = gaussian_blur_v(layer1)

    layer1_copy = layer1.copy()

    layer1_copy[:,:] = gradient[0, layer1[:,:, 2]]

    result = cv2.addWeighted(layer1_copy, 1, image, 1, 0)
    
    video_out.write(result)

    layer2.fill(0)
    cv2.putText(layer2, str(frame // FRAMERATE), (10, 100), 1, 2, WHITE)

    result = cv2.addWeighted(layer2, 1, result, 1, 0)

    # Comment or remove the following lines to prevent opening a window
    # cv2.imshow('Simulation', result)
    # key = cv2.waitKey(10)
    # if key == 27:  # Press 'Esc' to exit
    #    running = False

    # Add a condition to stop the loop if needed
    # Example: stop after a certain number of frames
    if frame > 100:  # adjust the frame limit as needed
        running = False

# Make sure to release the video file
cv2.destroyAllWindows()
video_out.release()
