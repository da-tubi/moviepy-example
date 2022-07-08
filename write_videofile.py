from moviepy.editor import *

clip = VideoFileClip("data/Mojito.flv").subclip(50,60)
clip.write_videofile("data/Mojito_10s.mp4")
