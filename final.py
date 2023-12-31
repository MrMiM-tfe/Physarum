import cv2
import numpy as np
import math

WIDTH, HEIGHT = 500, 500
AGENT_SIZE = 1
NUM_AGENTS = 1000

SPEED = 2  # pixel/frame
DIR_ANGLE = 20

SENSOR_ANGLE = 30

SENSOR_LENGTH = 10

EVAPORATION = 20

DIFFUSION = 0.5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED
        self.pheromone = np.random.randint(100, 200)

        center_x = WIDTH / 2
        center_y = HEIGHT / 2

        # to_center_vector = np.array([center_x - self.x, center_y - self.y])
        # normalized_vector = to_center_vector / np.linalg.norm(to_center_vector)
        # self.direction = normalized_vector * SPEED

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

def diffuse(img, diffuse_rate, decay_rate, delta_time):
    height, width, _ = img.shape

    # Create a padded image to handle boundary conditions
    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    # Create a 3x3 kernel for the blur
    kernel = np.ones((3, 3), dtype=np.float32) / 9

    # Apply the blur using filter2D
    blurred_img = cv2.filter2D(padded_img.astype(np.float32), -1, kernel)[1:-1, 1:-1]

    # Diffusion operation
    diffuse_weight = min(1.0, max(0.0, diffuse_rate * delta_time))
    blended_img = img * (1 - diffuse_weight) + blurred_img * diffuse_weight

    # Decay operation
    diffused_img = np.maximum(0, blended_img - decay_rate * delta_time)

    return diffused_img.astype(np.uint8)

# Example usage:
# Assuming img is your input image (numpy array) and other parameters are defined.
# diffused_image = diffuse_optimized(img, diffuse_rate, decay_rate, delta_time)


# agents = [Agent(np.random.randint(WIDTH), np.random.randint(HEIGHT)) for _ in range(NUM_AGENTS)]
agents = [Agent(np.random.randint(int(WIDTH / 2) - 100, int(WIDTH / 2) + 100), np.random.randint(int(HEIGHT / 2) - 100, int(HEIGHT / 2) + 100)) for _ in range(NUM_AGENTS)]

# Initialize image with three channels for BGR
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
layer1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Create a named window for visualization
cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Simulation', WIDTH + 400, HEIGHT + 400)  


# Set up VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (mp4v works well)
video_out = cv2.VideoWriter('simulation_video.mp4', fourcc, 30, (WIDTH, HEIGHT))

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
        image[agent.y, agent.x] = (255, 255, 255)
        # cv2.circle(image, (agent.x, agent.y), AGENT_SIZE, WHITE, -1)

    # layer1 = cv2.GaussianBlur(layer1, (3, 3), 0)
    layer1 = diffuse(layer1, 0.5, 0.1, 5)

    result = cv2.addWeighted(layer1, 1, image, 1, 0)

    # Display the image
    cv2.imshow('Simulation', result)
    video_out.write(result)

cv2.destroyAllWindows()
video_out.release()
