from moviepy.editor import *

images_folder = 'data/Mojito_10s'
clip = ImageSequenceClip(images_folder, fps=30)

clip.write_videofile('data/images_to_video.mp4')
