def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = SPEED # number 5
        self.pheromone = np.random.randint(255)

        # i: x , j: y 
        i = np.random.randint(-self.speed, self.speed)
        j = math.sqrt(self.speed**2 - i**2)
        self.direction = [i, j]

def move(self):
        self.x += self.direction[0]
        self.y += self.direction[1]

        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            self.direction = [-self.direction[0], -self.direction[1]]

            self.x += self.direction[0]
            self.y += self.direction[1]