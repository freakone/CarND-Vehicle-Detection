import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
import pickle
import os.path

class Detector():
    def __init__(self, svc):
        self.svc = svc

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def find_cars(self, img, ystart_stop, color, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hist_bins, channel):
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        ystart = ystart_stop[0]
        ystop = ystart_stop[1]
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = self.svc.convert_color(img_tosearch, color_space=color)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch[:,:,0].shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ctrans_tosearch[:,:,0].shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image

        hogs = []
        if channel == 'ALL':
            for ch in range(ctrans_tosearch.shape[2]):
                hogs.append(self.svc.get_hog_features(ctrans_tosearch[:,:,ch], orient, pix_per_cell, cell_per_block, feature_vec=False))      
        else:
            hogs.append(self.svc.get_hog_features(ctrans_tosearch[:,:,channel], orient, pix_per_cell, cell_per_block, feature_vec=False))      
        
        detections = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch

                hog_features = []
                for h in hogs:
                    hog_features.extend(h[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                hist_features = self.svc.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    rect_box = (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)
                    # cv2.rectangle(draw_img,rect_box,(0,0,255),6) 
                    detections.append(rect_box)    
                    
        return detections

    def overlay_detection(self, image, debug=False):
        windows = []
        boxes = []
        for scale in self.svc.params['scales']:
            boxes.extend(self.find_cars(image, self.svc.params['y_start_stop'], self.svc.params['color_space'], scale, self.svc.svc, 
                        self.svc.X_scaler, self.svc.params['orient'], self.svc.params['pix_per_cell'], self.svc.params['cell_per_block'], 
                        self.svc.params['hist_bins'], self.svc.params['hog_channel']))

        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = self.add_heat(heat,boxes)   
             
        heat = self.apply_threshold(heat, self.svc.params['heat_threshold'])
        heatmap = np.clip(heat, 0, 255)
        
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)

        if debug:
            raw_boxes = self.draw_boxes(image, boxes, color=(0, 0, 255), thick=6)   
            return raw_boxes, heatmap, draw_img

        return draw_img