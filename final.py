import cv2
import numpy as np
import math

WIDTH, HEIGHT = 200, 200
AGENT_SIZE = 1
NUM_AGENTS = 100

SPEED = 2  # pixel/frame
DIR_ANGLE = 30

SENSOR_ANGLE = 40

SENSOR_LENGTH = 5

EVAPORATION = 4

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

mymap = np.zeros((HEIGHT + 1, WIDTH + 1)).tolist()

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED
        self.pheromone = np.random.randint(100, 255)

        i = np.random.randint(-self.speed, self.speed)
        j = math.sqrt(self.speed**2 - i**2)
        self.direction = np.array([i, j])

    def move(self, image, image2):
        last_x = int(self.x)
        last_y = int(self.y)

        moving_dir = self.direction / np.linalg.norm(self.direction)
        sensor_dir = moving_dir * SENSOR_LENGTH

        # Add red sensors to detect red areas
        front_sensor_x = int(self.x + sensor_dir[0])
        front_sensor_y = int(self.y + sensor_dir[1])

        left_sensor_x = int(self.x + sensor_dir[0] * math.cos(math.radians(SENSOR_ANGLE)) - sensor_dir[1] * math.sin(math.radians(SENSOR_ANGLE)))
        left_sensor_y = int(self.y + sensor_dir[1] * math.cos(math.radians(SENSOR_ANGLE)) + sensor_dir[0] * math.sin(math.radians(SENSOR_ANGLE)))

        right_sensor_x = int(self.x + sensor_dir[0] * math.cos(math.radians(-SENSOR_ANGLE)) - sensor_dir[1] * math.sin(math.radians(-SENSOR_ANGLE)))
        right_sensor_y = int(self.y + sensor_dir[1] * math.cos(math.radians(-SENSOR_ANGLE)) + sensor_dir[0] * math.sin(math.radians(-SENSOR_ANGLE)))

        try:
            image2[front_sensor_y, front_sensor_x] = (0, 100, 100)
            image2[left_sensor_y, left_sensor_x] = (0, 100, 100)
            image2[right_sensor_y, right_sensor_x] = (0, 100, 100)

            front_sensor_value = image[front_sensor_y, front_sensor_x, 2]
            left_sensor_value = image[left_sensor_y, left_sensor_x, 2]
            right_sensor_value = image[right_sensor_y, right_sensor_x, 2]

            sens = 1

            # Adjust direction based on red sensor values
            if front_sensor_value > left_sensor_value and front_sensor_value > right_sensor_value or (right_sensor_value < sens and left_sensor_value < sens):
                # Move straight towards the area with the highest red intensity
                # self.direction = [self.direction[0], self.direction[1]]
                pass
            elif left_sensor_value > right_sensor_value:
                # Turn left towards the area with higher red intensity
                self.direction = [
                    self.direction[0] * math.cos(math.radians(-DIR_ANGLE)) - self.direction[1] * math.sin(math.radians(-DIR_ANGLE)),
                    self.direction[1] * math.cos(math.radians(-DIR_ANGLE)) + self.direction[0] * math.sin(math.radians(-DIR_ANGLE))
                ]
            else:
                # Turn right towards the area with higher red intensity
                self.direction = [
                    self.direction[0] * math.cos(math.radians(DIR_ANGLE)) - self.direction[1] * math.sin(math.radians(DIR_ANGLE)),
                    self.direction[1] * math.cos(math.radians(DIR_ANGLE)) + self.direction[0] * math.sin(math.radians(DIR_ANGLE))
                ]

            
        except:
            pass

        self.x = int(self.x + self.direction[0])
        self.y = int(self.y + self.direction[1])

        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            self.direction = [-self.direction[0], -self.direction[1]]

            self.x = int(self.x + self.direction[0])
            self.y = int(self.y + self.direction[1])

        path_pixels = get_line_pixels(last_x, last_y, self.x, self.y)

        # size = 10
        pheromone_list = []
        for path_pixel in path_pixels:
            # x, y = path_pixel
            pheromone = image[path_pixel[1], path_pixel[0]]
            _pheromone = pheromone[2] + self.pheromone
            pheromone[2] = _pheromone if _pheromone <= 255 else 255
            pheromone_list.append(pheromone)  # BGR format for OpenCV

        return (path_pixels, pheromone_list)

def red_filter(image):
    # Extract the red channel from the image
    red_channel = image[:, :, 2]

    # Get coordinates where the red component is greater than 0
    red_pixel_coordinates = np.column_stack(np.where(red_channel > 0))

    return red_pixel_coordinates

def simple_blur(image, x, y, size):

    region = image[max(0, y - size // 2):min(HEIGHT, y + size // 2),
                   max(0, x - size // 2):min(WIDTH, x + size // 2), :]
    average_color = np.mean(region, axis=(0, 1)) * 10
    # print (average_color)
    image[y - size // 2:y + size // 2, x - size // 2:x + size // 2, :] = average_color

    return image

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

    for red_pixel in red_pixels:
        pheromone = layer1[red_pixel[0], red_pixel[1]]
        _pheromone = pheromone[2] - EVAPORATION
        pheromone[2] = _pheromone if _pheromone >= 0 else 0
        layer1[red_pixel[0], red_pixel[1]] = pheromone

    if frame > 2 :
        layer1 = cv2.GaussianBlur(layer1, (5, 5), 1)
        frame = 0


    for agent in agents:
        path, pheromones = agent.move(layer1, image)

        for coordinate, pheromone in zip(path, pheromones):
            # Draw pheromone
            layer1[coordinate[1], coordinate[0]] = pheromone

        # Draw agent
        # cv2.circle(image, (agent.x, agent.y), AGENT_SIZE, WHITE, -1)

    result = cv2.addWeighted(layer1, 1, image, 1, 0)

    # Display the image
    cv2.imshow('Simulation', result)

cv2.destroyAllWindows()
