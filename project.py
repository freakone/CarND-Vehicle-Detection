import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from svc import SVC
from detector import Detector
from moviepy.editor import VideoFileClip

params = {  'color_space': 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
            'orient': 9,  # HOG orientations
            'pix_per_cell': 8, # HOG pixels per cell
            'cell_per_block': 2, # HOG cells per block
            'hog_channel': 'ALL',# Can be 0, 1, 2, or "ALL"
            'hist_bins': 32,  # Number of histogram bins
            'y_start_stop': [400, 700], # Min and max in y to search in slide_window()
            'heat_threshold': 2,
            'scales': [1, 1.5]
         }

svc = SVC(params)
detector = Detector(svc)


clip1 = VideoFileClip("project_video.mp4") 
white_clip = clip1.fl_image(detector.overlay_detection)
white_clip.write_videofile('output_images/project_video.mp4', audio=False)


# clip1 = VideoFileClip("test_video.mp4") 
# white_clip = clip1.fl_image(detector.overlay_detection)
# white_clip.write_videofile('output_images/test_video.mp4', audio=False)

# fig = plt.figure()

# for i in range(4):
#     image = mpimg.imread('test_images/test{}.jpg'.format(i+1))
#     raw_boxes, heatmap, draw_img = detector.overlay_detection(image, debug=True)

#     a=fig.add_subplot(4,3,(i*3+1))
#     plt.imshow(raw_boxes)
#     a=fig.add_subplot(4,3,(i*3+2))
#     plt.imshow(heatmap)
#     a=fig.add_subplot(4,3,(i*3+3))
#     plt.imshow(draw_img)

# fig.show()
# input()

# image = mpimg.imread('test_images/vlc.jpg')
# raw_boxes, heatmap, draw_img = detector.overlay_detection(image, debug=True)
# plt.imshow(draw_img)
# plt.show()