import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from svc import SVC
from detector import Detector

params = {  'color_space': 'RGB', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
            'orient': 9,  # HOG orientations
            'pix_per_cell': 16, # HOG pixels per cell
            'cell_per_block': 4, # HOG cells per block
            'hog_channel': 'ALL',# Can be 0, 1, 2, or "ALL"
            'spatial_size': (16, 16), # Spatial binning dimensions
            'hist_bins': 32,  # Number of histogram bins
            'spatial_feat': True, # Spatial features on or off
            'hist_feat': True, # Histogram features on or off
            'hog_feat': True, # HOG features on or off
            'y_start_stop': [400, 700] # Min and max in y to search in slide_window()
         }

svc = SVC(params)
detector = Detector(svc)

image = mpimg.imread('test_images/test1.jpg')
image = detector.overlay_detection(image)

# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(image)
plt.show()

