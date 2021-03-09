# importing the module  
import pytube
from pytube import YouTube  
print(pytube.__version__)
# where to save  
#SAVE_PATH = "/Users/zhampela/lab41/projects/FakeFinder/ff-dash-app/data/" 
SAVE_PATH = "/home/ubuntu/FakeFinder/dash/data/"
  
# link of the video to be downloaded  
link = "https://www.youtube.com/watch?v=2svOtXaD3gg&t=40s&ab_channel=CtrlShiftFace"
  
YouTube(link).streams.first().download()
  
# filters out all the files with "mp4" extension
#mp4files = yt.filter('mp4')
#mp4files = yt.streams.first()
  
#to set the name of the file 
#yt.set_filename('home_stallone')
  
# get the video with the extension and 
# resolution passed in the get() function  
#d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution)  
#try:  
#    # downloading the video  
#    d_video.download(SAVE_PATH)  
#except:  
#    print("Some Error!")  
#print('Task Completed!') 
