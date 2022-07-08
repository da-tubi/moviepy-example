from moviepy.editor import *
import os

output_folder = "data/Mojito_10s"
os.mkdir(output_folder)

clip = VideoFileClip("data/Mojito.flv").subclip(50,60)
clip.write_images_sequence(f"{output_folder}/frame%03d.png")
