import cv2
import numpy as np

def create_gradient(colors):
    image_size = (255,1)
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

colors = [(255, 0, 16), (200, 0, 100), (0, 0, 163)]

# Create the gradient image
gradient_image = create_gradient(colors)

# Display the gradient image
print(gradient_image)
cv2.imshow('Gradient Image', gradient_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
