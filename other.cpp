#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

const int WIDTH = 30;
const int HEIGHT = 30;
const int NUM_AGENTS = 10;
const int SPEED = 2;
const int DIR_ANGLE = 30;
const int SENSOR_ANGLE = 40;
const int SENSOR_LENGTH = 5;
const int EVAPORATION = 4;

constexpr double DEG_TO_RAD = M_PI / 180.0;

class Agent {
public:
    int x, y;
    double speed;
    int pheromone;
    double direction;

    Agent(int _x, int _y) : x(_x), y(_y), speed(SPEED), pheromone(rand() % 156 + 100) {
        direction = rand() % 360;
    }

    std::vector<std::pair<int, int>> move(std::vector<std::vector<int>>& environment);
};

std::vector<std::pair<int, int>> Agent::move(std::vector<std::vector<int>>& environment) {
    int last_x = x;
    int last_y = y;

    double rad_direction = direction * DEG_TO_RAD;
    double sensor_x = x + SENSOR_LENGTH * cos(rad_direction);
    double sensor_y = y + SENSOR_LENGTH * sin(rad_direction);

    // Handle sensor collisions with the environment bounds
    if (sensor_x < 0 || sensor_x >= WIDTH || sensor_y < 0 || sensor_y >= HEIGHT) {
        direction += 180; // Turn around if hitting the bounds
    } else {
        int sensor_value = environment[static_cast<int>(sensor_y)][static_cast<int>(sensor_x)];

        if (sensor_value > 0) {
            // Move straight towards the area with the highest intensity
        } else {
            // Turn left towards the area with higher intensity
            direction += DIR_ANGLE;
        }
    }

    // Update agent position
    x = static_cast<int>(x + speed * cos(rad_direction));
    y = static_cast<int>(y + speed * sin(rad_direction));

    if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) {
        direction += 180; // Turn around if hitting the bounds
        x = static_cast<int>(x + speed * cos(rad_direction));
        y = static_cast<int>(y + speed * sin(rad_direction));
    }

    std::vector<std::pair<int, int>> path_pixels;

    // Record agent path
    for (int i = 0; i <= speed; ++i) {
        int path_x = last_x + static_cast<int>(i * cos(rad_direction));
        int path_y = last_y + static_cast<int>(i * sin(rad_direction));
        path_pixels.emplace_back(path_x, path_y);
    }

    return path_pixels;
}

void printEnvironment(const std::vector<std::vector<int>>& environment) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            if (environment[i][j] > 0) {
                std::cout << "X";
            } else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<Agent> agents;
    agents.reserve(NUM_AGENTS);

    for (int i = 0; i < NUM_AGENTS; ++i) {
        agents.emplace_back(rand() % WIDTH, rand() % HEIGHT);
    }

    std::vector<std::vector<int>> environment(HEIGHT, std::vector<int>(WIDTH, 0));

    bool running = true;
    int frame = 0;

    while (running) {
        ++frame;

        // EVAPORATION
        for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                if (environment[i][j] > 0) {
                    environment[i][j] -= EVAPORATION;
                    environment[i][j] = (environment[i][j] >= 0) ? environment[i][j] : 0;
                }
            }
        }

        for (auto& agent : agents) {
            std::vector<std::pair<int, int>> path = agent.move(environment);

            // Update environment with agent path
            for (const auto& pixel : path) {
                environment[pixel.second][pixel.first] = agent.pheromone;
            }
        }

        // Print environment (simple ASCII art representation)
        printEnvironment(environment);

        if (frame >= 100) {
            // Add delay or other processing as needed
            frame = 0;
        }
    }

    return 0;
}
