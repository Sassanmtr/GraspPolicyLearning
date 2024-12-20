import os
from pathlib import Path
from PIL import Image

# Directory containing the PNG images
directory = Path.cwd() / "collected_data" / "traj3" / "rgb"
# Get all PNG files in the directory
image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".png")]
# Sort the image paths using a numerical sorting key
image_paths.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
# Open each image and append to a list
images = []
for path in image_paths:
    image = Image.open(path)
    images.append(image)
# Resize the images if desired
# Example: Resize to a width of 800 pixels while preserving the aspect ratio
# images = [image.resize((800, int(image.size[1] * 800 / image.size[0]))) for image in images]

# Create the GIF with increased duration, quality, and optimization
output_path = Path.cwd() / "collected_data" / "traj3" / "output.gif"
images[0].save(output_path, save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0, quality=95)
