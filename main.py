import pygame
import numpy as np
import math
from get_line_pixels import get_line_pixels
from red_filter import red_filter

WIDTH, HEIGHT = 800, 600
AGENT_SIZE = 4
NUM_AGENTS = 10

SPEED = 8 # pixel/frame

EVAPORATION = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

mymap = np.zeros((HEIGHT + 1, WIDTH + 1)).tolist()

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED
        self.pheromone = np.random.randint(255)

        i = np.random.randint(-self.speed, self.speed)
        j = math.sqrt(self.speed**2 - i**2)
        self.direction = [i, j]

    def move(self, layer: pygame.Surface):

        last_x = int(self.x)
        last_y = int(self.y)

        self.x = int(self.x + self.direction[0])
        self.y = int(self.y + self.direction[1])

        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            self.direction = [-self.direction[0], -self.direction[1]]

            self.x = int(self.x + self.direction[0])
            self.y = int(self.y + self.direction[1])

        path_pixels = get_line_pixels(last_x, last_y, self.x, self.y)

        pheromone_list = []
        for path_pixel in path_pixels:
            pheromone = layer.get_at((path_pixel[0], path_pixel[1]))
            _pheromone = pheromone[0] + self.pheromone
            pheromone[0] = _pheromone if _pheromone <= 255 else 255
            pheromone_list.append(pheromone)

        return (path_pixels, pheromone_list)

agents = [Agent(np.random.randint(WIDTH), np.random.randint(HEIGHT)) for _ in range(NUM_AGENTS)]

# init pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

layer1 = pygame.surface.Surface((WIDTH, HEIGHT))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    red_pixels = red_filter(layer1)
    for red_pixel in red_pixels:
        pheromone = layer1.get_at(red_pixel)
        _pheromone = pheromone[0] - EVAPORATION
        pheromone[0] = _pheromone if _pheromone >= 0 else 0
        pygame.draw.circle(layer1, pheromone, red_pixel, 1)
    
    for agent in agents:
        path, pheromones = agent.move(layer1)

        for coordinate, pheromone in zip(path, pheromones):
            # draw pheromone
            pygame.draw.circle(layer1, pheromone, coordinate, 1)

        # draw agent
        pygame.draw.circle(screen, WHITE, (agent.x, agent.y), AGENT_SIZE)
    
    screen.blit(layer1,(0,0), special_flags=pygame.BLEND_ADD)

    pygame.display.flip()
    pygame.time.delay(10)