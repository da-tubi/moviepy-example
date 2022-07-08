from moviepy.editor import *
from rikai.types import Image

images_folder = 'data/Mojito_10s'
images = []
for png in sorted(os.listdir(images_folder)):
    full_png = f"{images_folder}/{png}"
    image = Image(full_png)
    images.append(image.to_numpy())

clip = ImageSequenceClip(images, fps=30)

clip.write_videofile('data/images_to_video2.mp4')
