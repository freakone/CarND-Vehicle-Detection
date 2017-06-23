import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from svc import SVC
from detector import Detector
from moviepy.editor import VideoFileClip

params = {  'color_space': 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
            'orient': 11,  # HOG orientations
            'pix_per_cell': 12, # HOG pixels per cell
            'cell_per_block': 2, # HOG cells per block
            'hog_channel': 'ALL',# Can be 0, 1, 2, or "ALL"
            'spatial_size': (16, 16), # Spatial binning dimensions
            'hist_bins': 32,  # Number of histogram bins
            'spatial_feat': False, # Spatial features on or off
            'hist_feat': True, # Histogram features on or off
            'hog_feat': True, # HOG features on or off
            'y_start_stop': [400, 700], # Min and max in y to search in slide_window()
            'heat_threshold': 4,
            'window_size': (80, 80),
            'scales': [1, 1.5, 2, 2.5]

         }

svc = SVC(params)
detector = Detector(svc)

clip1 = VideoFileClip("test_video.mp4") 
white_clip = clip1.fl_image(detector.overlay_detection)
white_clip.write_videofile('output_images/test_video.mp4', audio=False)

# image = mpimg.imread('test_images/vlc2.jpg')
# image = detector.overlay_detection(image)
# plt.imshow(image)
# plt.show()

