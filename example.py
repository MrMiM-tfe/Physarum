import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class ImageDisplayApp:
    def __init__(self, root, image_array):
        self.root = root
        self.root.title("Image Display App")

        # Convert NumPy array to PIL Image
        self.image = Image.fromarray(image_array)

        # Create Tkinter PhotoImage object from PIL Image
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Create a label to display the image
        self.image_label = tk.Label(root, image=self.tk_image)
        self.image_label.pack()

        # Add a button to perform some action (you can customize this)
        self.button = tk.Button(root, text="Click me!", command=self.button_click)
        self.button.pack()

    def button_click(self):
        # Example action to perform when the button is clicked
        print("Button clicked!")

# Example NumPy image array (replace this with your own image data)
# You may need to adjust the array dimensions and data type based on your image.
image_array = np.random.randint(0, 255, size=(300, 400, 3), dtype=np.uint8)

# Create the Tkinter root window
root = tk.Tk()

# Create the ImageDisplayApp instance
app = ImageDisplayApp(root, image_array)

# Run the Tkinter event loop
root.mainloop()
