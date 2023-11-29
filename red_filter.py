import pygame
import numpy as np

def red_filter(screen):
    """
    Get coordinates of red pixels from a Pygame screen where the red component is greater than 0.

    Parameters:
    - screen: Pygame screen.

    Returns:
    - red_pixel_coordinates: List of (x, y) coordinates of red pixels.
    """

    # Capture the screen as a NumPy array
    pixels = pygame.surfarray.array3d(screen)

    # Get coordinates where the red component is greater than 0
    red_pixel_coordinates = np.column_stack(np.where(pixels[:, :, 0] > 0))

    return red_pixel_coordinates
