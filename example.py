import pygame
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 10
NUM_CELLS_X = WIDTH // GRID_SIZE
NUM_CELLS_Y = HEIGHT // GRID_SIZE
NUM_AGENTS = 10
MEMORY_DECAY = 0.9  # Memory decay factor

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Agent class
class PhysarumAgent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path = [(x, y)]
        self.alpha_values = [255]
        self.prev_direction = np.array([0, 0])

    def move(self):
        # Physarum movement rules
        neighbors = get_neighbors(self.x, self.y)
        if neighbors:
            prev_direction = self.prev_direction
            # Introduce bias towards the previous direction
            if np.random.rand() < 0.8:
                neighbors = sorted(neighbors, key=lambda pos: np.dot(np.array(pos) - np.array([self.x, self.y]), prev_direction))

            idx = np.random.choice(len(neighbors))
            new_x, new_y = neighbors[idx]
            self.prev_direction = np.array([new_x - self.x, new_y - self.y])
            self.x, self.y = new_x, new_y
            self.path.append((new_x, new_y))
            self.alpha_values.append(255)

    def update_alpha_values(self):
        self.alpha_values = [max(0, alpha - 5) for alpha in self.alpha_values]

# Function to get neighboring cells
def get_neighbors(x, y):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_x, new_y = x + i, y + j
            if 0 <= new_x < NUM_CELLS_X and 0 <= new_y < NUM_CELLS_Y:
                neighbors.append((new_x, new_y))
    return neighbors

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physarum Simulation")

# Create agents
agents = [PhysarumAgent(np.random.randint(NUM_CELLS_X), np.random.randint(NUM_CELLS_Y)) for _ in range(NUM_AGENTS)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move agents
    for agent in agents:
        agent.move()
        agent.update_alpha_values()

    # Draw agents and their paths with fading effect
    screen.fill(BLACK)
    for agent in agents:
        for (x, y), alpha in zip(agent.path, agent.alpha_values):
            color = (alpha, 0, 0)
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        x, y = agent.path[-1]
        pygame.draw.rect(screen, WHITE, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()
