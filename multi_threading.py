import cv2
import numpy as np
import math
import os
import threading


WIDTH, HEIGHT = 500, 500
AGENT_SIZE = 1
NUM_AGENTS = 3000

AGENTS = False
IN_CENTER = False

SPEED = 2  # pixel/frame
DIR_ANGLE = 20 # deg

SENSOR_ANGLE = 30 # deg
SENSOR_LENGTH = 10 # pixel

PHEROMONE_MIN = 50 # 0
PHEROMONE_MAX = 160 # 255

EVAPORATION = 4 # [0, 255]
# DIFFUSION = 0.5
BLUR_SCALE = 3
NUM_BLUR_THREADS = 5

WOBBLING = 10
WOBBLING_CHANCE = 30 # [0, 100]

SCALE = 1.5
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

            # Adjust direction based on red sensor values
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
        except IndexError:
            pass

        self.x = int(self.x + self.direction[0])
        self.y = int(self.y + self.direction[1])

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

        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            self.direction *= -1

            self.x = int(self.x + self.direction[0])
            self.y = int(self.y + self.direction[1])

        path_pixels = get_line_pixels(last_x, last_y, self.x, self.y)

        pheromone_list = []
        for pixel in path_pixels:
            x, y = pixel
            pheromone = image[y, x][2] + self.pheromone
            image[y, x][2] = min(255, pheromone)
            pheromone_list.append(image[y, x])

        return (path_pixels, pheromone_list)

def red_filter(image):
    # Extract the red channel from the image
    red_channel = image[:, :, 2]

    # Get coordinates where the red component is greater than 0
    red_pixel_coordinates = np.column_stack(np.where(red_channel > 0))

    return red_pixel_coordinates

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


if IN_CENTER:
    agents = [Agent(np.random.randint(int(WIDTH / 2) - 100, int(WIDTH / 2) + 100), np.random.randint(int(HEIGHT / 2) - 100, int(HEIGHT / 2) + 100)) for _ in range(NUM_AGENTS)]
else:
    agents = [Agent(np.random.randint(WIDTH), np.random.randint(HEIGHT)) for _ in range(NUM_AGENTS)]

# Initialize image with three channels for BGR
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Create a named window for visualization
cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Simulation', int(WIDTH * SCALE), int(HEIGHT * SCALE))


# Set up VideoWriter
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

running = True
frame = 0
while running:
    frame += 1
    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        running = False

    image.fill(0)  # Clear the image

    # EVAPORATION
    red_pixels = red_filter(layer1)

    red_pixels = np.array(red_pixels)
    layer1_red_pixels = layer1[red_pixels[:, 0], red_pixels[:, 1]]

    # mask = np.logical_not(np.logical_and(np.arange(3)[:, None] == 1, np.arange(3) == 1))
    # for x, y in red_pixels:
    #     layer1[x - 1:x + 1, y - 1:y + 1, 2] = np.minimum(255, (layer1[x,y,2] * 0.5) + layer1[x - 1:x + 1, y - 1:y + 1, 2])

    # print(red_pixels[:, 0] - 1, end=" ")
    # print(red_pixels[:, 1])
    # layer1[red_pixels[:, 0] - 1:red_pixels[:, 0] + 1, red_pixels[:, 1] - 1:red_pixels[:, 1] + 1][1] = 255

    # Update pheromone values using vectorized operations
    layer1_red_pixels[:, 2] = np.maximum(0, layer1_red_pixels[:, 2] * 0.9)

    layer1[red_pixels[:, 0], red_pixels[:, 1]] = layer1_red_pixels


    for agent in agents:
        path, pheromones = agent.move(layer1, image)

        for coordinate, pheromone in zip(path, pheromones):
            # Draw pheromone
            layer1[coordinate[1], coordinate[0]] = pheromone

        # Draw agent
        if AGENTS:
            image[agent.y, agent.x] = (255, 255, 255)
        # cv2.circle(image, (agent.x, agent.y), AGENT_SIZE, WHITE, -1)

    layer1 = gaussian_blur_h(layer1)
    layer1 = gaussian_blur_v(layer1)
    # layer1 = parallel_diffuse(layer1, 0.5, 0.1, 5)

    result = cv2.addWeighted(layer1, 1, image, 1, 0)

    video_out.write(result)

    layer2.fill(0)
    cv2.putText(layer2, str(frame // FRAMERATE), (10, 100), 1, 2, WHITE)

    result = cv2.addWeighted(layer2, 1, result, 1, 0)
    # Display the image
    cv2.imshow('Simulation', result)

cv2.destroyAllWindows()
video_out.release()
