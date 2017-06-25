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
import matplotlib.image as mpimg
import numpy as np
import cv2
import os.path

class SVC():
    def __init__(self, params):
        self.params = params
        self.filename = "svc.p"
        self.load_svc()
        
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block), 
                                    transform_sqrt=True, 
                                    visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        else:      
            features = hog(img, orientations=orient, 
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), 
                        transform_sqrt=False, 
                        visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, size=(32, 32)):        
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return features

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        hist_feat=True, hog_feat=True):
        features = []
        for img in imgs:  
            img = mpimg.imread(img)          
            features.append(self.single_img_features(img, color_space, spatial_size, hist_bins, 
                            orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat))
        return features

    def convert_color(self, img, color_space='YCrCb'):
        if color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        if color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        hist_feat=True, hog_feat=True):    
        img_features = []
        if color_space != 'RGB':
            feature_image = self.convert_color(img, color_space=color_space)
        else: feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            # img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    def load_svc(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                (self.svc, self.X_scaler) = pickle.load(f)
        else:
            self.train()

    def train(self):
        # Read in cars and notcars
        cars = glob.glob('vehicles/*/*.png')
        notcars = glob.glob('non-vehicles/*/*.png')

        car_features = self.extract_features(cars, color_space=self.params['color_space'], 
                                            spatial_size=self.params['spatial_size'], hist_bins=self.params['hist_bins'], 
                                            orient=self.params['orient'], pix_per_cell=self.params['pix_per_cell'], 
                                            cell_per_block=self.params['cell_per_block'], 
                                            hog_channel=self.params['hog_channel'])
        notcar_features = self.extract_features(notcars, color_space=self.params['color_space'], 
                                                spatial_size=self.params['spatial_size'], hist_bins=self.params['hist_bins'], 
                                                orient=self.params['orient'], pix_per_cell=self.params['pix_per_cell'], 
                                                cell_per_block=self.params['cell_per_block'], 
                                                hog_channel=self.params['hog_channel'])
                
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',self.params['orient'],'orientations',self.params['pix_per_cell'],
            'pixels per cell and', self.params['cell_per_block'],'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()

        with open(self.filename, 'wb') as f:
            pickle.dump((self.svc, self.X_scaler), f)