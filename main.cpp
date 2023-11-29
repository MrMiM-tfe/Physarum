#define CVAPI_EXPORTS
#include <opencv2/opencv.hpp>
#include <mutex>
#include <iostream>
#include <vector>
#include <cmath>
#include <cmath>

const int WIDTH = 300;
const int HEIGHT = 300;
const int AGENT_SIZE = 1;
const int NUM_AGENTS = 1000;
const int SPEED = 2;
const int DIR_ANGLE = 30;
const int SENSOR_ANGLE = 40;
const int SENSOR_LENGTH = 5;
const int EVAPORATION = 4;

cv::Scalar WHITE = cv::Scalar(255, 255, 255);
cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar RED = cv::Scalar(255, 0, 0);

constexpr double DEG_TO_RAD = CV_PI / 180.0;  

std::vector<cv::Point> get_line_pixels(int x1, int y1, int x2, int y2) {
    std::vector<cv::Point> pixels;

    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (x1 != x2 || y1 != y2) {
        pixels.push_back(cv::Point(x1, y1));
        int e2 = 2 * err;

        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }

        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }

    return pixels;
}

std::vector<cv::Point> red_filter(const cv::Mat& image) {
    std::vector<cv::Point> red_pixel_coordinates;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.at<cv::Vec3b>(y, x)[2] > 0) {
                red_pixel_coordinates.push_back(cv::Point(x, y));
            }
        }
    }

    return red_pixel_coordinates;
}

class Agent {
public:
    int x, y;
    double speed;
    int pheromone;
    cv::Vec2d direction;

    Agent(int _x, int _y) : x(_x), y(_y), speed(SPEED), pheromone(rand() % 156 + 100) {
        int i = rand() % (2 * static_cast<int>(speed)) - static_cast<int>(speed);
        int j = sqrt(speed * speed - i * i);
        direction = cv::Vec2d(i, j);
    }

    std::pair<std::vector<cv::Point>, std::vector<cv::Vec3b>> move(cv::Mat& image, cv::Mat& image2);
};

std::pair<std::vector<cv::Point>, std::vector<cv::Vec3b>> Agent::move(cv::Mat& image, cv::Mat& image2) {
    int last_x = static_cast<int>(x);
    int last_y = static_cast<int>(y);

    cv::Vec2d moving_dir = direction / cv::norm(direction);
    cv::Vec2d sensor_dir = moving_dir * SENSOR_LENGTH;

    int front_sensor_x = static_cast<int>(x + sensor_dir[0]);
    int front_sensor_y = static_cast<int>(y + sensor_dir[1]);

    int left_sensor_x = static_cast<int>(x + sensor_dir[0] * cos(SENSOR_ANGLE * DEG_TO_RAD) - sensor_dir[1] * sin(SENSOR_ANGLE * DEG_TO_RAD));
    int left_sensor_y = static_cast<int>(y + sensor_dir[1] * cos(SENSOR_ANGLE * DEG_TO_RAD) + sensor_dir[0] * sin(SENSOR_ANGLE * DEG_TO_RAD));

    int right_sensor_x = static_cast<int>(x + sensor_dir[0] * cos(-SENSOR_ANGLE * DEG_TO_RAD) - sensor_dir[1] * sin(-SENSOR_ANGLE * DEG_TO_RAD));
    int right_sensor_y = static_cast<int>(y + sensor_dir[1] * cos(-SENSOR_ANGLE * DEG_TO_RAD) + sensor_dir[0] * sin(-SENSOR_ANGLE * DEG_TO_RAD));

    try {
        int front_sensor_value = image.at<cv::Vec3b>(front_sensor_y, front_sensor_x)[2];
        int left_sensor_value = image.at<cv::Vec3b>(left_sensor_y, left_sensor_x)[2];
        int right_sensor_value = image.at<cv::Vec3b>(right_sensor_y, right_sensor_x)[2];

        int sens = 1;

        if (front_sensor_value > left_sensor_value && front_sensor_value > right_sensor_value || (right_sensor_value < sens && left_sensor_value < sens)) {
            // Move straight towards the area with the highest red intensity
        } else if (left_sensor_value > right_sensor_value) {
            // Turn left towards the area with higher red intensity
            direction = cv::Vec2d(
                direction[0] * cos(-DIR_ANGLE * DEG_TO_RAD) - direction[1] * sin(-DIR_ANGLE * DEG_TO_RAD),
                direction[1] * cos(-DIR_ANGLE * DEG_TO_RAD) + direction[0] * sin(-DIR_ANGLE * DEG_TO_RAD)
            );
        } else {
            // Turn right towards the area with higher red intensity
            direction = cv::Vec2d(
                direction[0] * cos(DIR_ANGLE * DEG_TO_RAD) - direction[1] * sin(DIR_ANGLE * DEG_TO_RAD),
                direction[1] * cos(DIR_ANGLE * DEG_TO_RAD) + direction[0] * sin(DIR_ANGLE * DEG_TO_RAD)
            );
        }
    } catch (const cv::Exception& e) {
        // Handle the exception
    }

    x = static_cast<int>(x + direction[0]);
    y = static_cast<int>(y + direction[1]);

    if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) {
        direction = cv::Vec2d(-direction[0], -direction[1]);

        x = static_cast<int>(x + direction[0]);
        y = static_cast<int>(y + direction[1]);
    }

    std::vector<cv::Point> path_pixels = get_line_pixels(last_x, last_y, x, y);
    std::vector<cv::Vec3b> pheromone_list;

    for (const auto& path_pixel : path_pixels) {
        cv::Vec3b pheromone = image.at<cv::Vec3b>(path_pixel.y, path_pixel.x);
        int _pheromone = pheromone[2] + pheromone[2];
        pheromone[2] = (_pheromone <= 255) ? _pheromone : 255;
        pheromone_list.push_back(pheromone);
    }

    return std::make_pair(path_pixels, pheromone_list);
}



int main() {
    std::vector<Agent> agents;
    agents.reserve(NUM_AGENTS);

    for (int i = 0; i < NUM_AGENTS; ++i) {
        agents.emplace_back(rand() % WIDTH, rand() % HEIGHT);
    }

    cv::Mat image(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat layer1(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::namedWindow("Simulation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Simulation", WIDTH + 400, HEIGHT + 400);

    bool running = true;
    int frame = 0;

    while (running) {
        ++frame;
        int key = cv::waitKey(10);

        if (key == 27) {  // Press 'Esc' to exit
            running = false;
        }

        image = cv::Scalar(0, 0, 0);  // Clear the image

        // EVAPORATION
        std::vector<cv::Point> red_pixels = red_filter(layer1);

        for (const auto& red_pixel : red_pixels) {
            cv::Vec3b pheromone = layer1.at<cv::Vec3b>(red_pixel.y, red_pixel.x);
            int _pheromone = pheromone[2] - EVAPORATION;
            pheromone[2] = (_pheromone >= 0) ? _pheromone : 0;
            layer1.at<cv::Vec3b>(red_pixel.y, red_pixel.x) = pheromone;
        }

        if (frame >= 0) {
            cv::GaussianBlur(layer1, layer1, cv::Size(5, 5), 1);
            frame = 0;
        }

        for (auto& agent : agents) {
            std::pair<std::vector<cv::Point>, std::vector<cv::Vec<unsigned char, 3>>> result = agent.move(layer1, image);
            std::vector<cv::Point> path = result.first;
            std::vector<cv::Vec<unsigned char, 3>> pheromones = result.second;


            for (size_t i = 0; i < path.size(); ++i) {
                // Draw pheromone
                layer1.at<cv::Vec3b>(path[i].y, path[i].x) = pheromones[i];
            }
        }

        cv::Mat result;
        cv::addWeighted(layer1, 1, image, 1, 0, result);

        cv::imshow("Simulation", result);
    }

    cv::destroyAllWindows();

    return 0;
}

