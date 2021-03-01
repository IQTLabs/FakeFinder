import ffmpeg_streaming
from ffmpeg_streaming import Formats

fname = '/Users/zhampela/lab41/projects/FakeFinder/ff-dash-app/data/Home-Stallone-[DeepFake].mp4'

video = ffmpeg_streaming.input(fname)

dash = video.dash(Formats.h264())
dash.auto_generate_representations()

dash.output()
#dash.output('/var/media/dash.mpd')
