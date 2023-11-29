import cv2
import numpy as np
import math

WIDTH, HEIGHT = 300, 300
AGENT_SIZE = 1
NUM_AGENTS = 1000

SPEED = 2  # pixel/frame
DIR_ANGLE = 30

SENSOR_ANGLE = 30

SENSOR_LENGTH = 8

EVAPORATION = 4

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED
        self.pheromone = np.random.randint(100, 200)

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

agents = [Agent(np.random.randint(WIDTH), np.random.randint(HEIGHT)) for _ in range(NUM_AGENTS)]

# Initialize image with three channels for BGR
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Create a named window for visualization
cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Simulation', WIDTH + 400, HEIGHT + 400)  

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

    # Update pheromone values using vectorized operations
    layer1_red_pixels[:, 2] = np.maximum(0, layer1_red_pixels[:, 2] - EVAPORATION)

    layer1[red_pixels[:, 0], red_pixels[:, 1]] = layer1_red_pixels


    for agent in agents:
        path, pheromones = agent.move(layer1, image)

        for coordinate, pheromone in zip(path, pheromones):
            # Draw pheromone
            layer1[coordinate[1], coordinate[0]] = pheromone

        # Draw agent
        # cv2.circle(image, (agent.x, agent.y), AGENT_SIZE, WHITE, -1)
    layer1 = cv2.GaussianBlur(layer1, (5, 5), 1)

    result = cv2.addWeighted(layer1, 1, image, 1, 0)

    # Display the image
    cv2.imshow('Simulation', result)

cv2.destroyAllWindows()
