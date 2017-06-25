# Vehicle Detection Project
## Kamil Górski

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./output_images/figure_1.png
[detection]: ./output_images/figure_2.png
[heatmap]: ./output_images/figure_3.png
[pipeline_example]: ./output_images/figure_4.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `svc.py` file, main training function is between the `120` and `166` line. HOG feature extraction is in lines `93-104`.

I have read all the training images into two subsets: `non-vehicles` and `vehicles`.

Then I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented a lot of combinations of parameters: color space, orientation, pixels per cell and cell per block. I assumed that HOG channel will be always set to "ALL". The main aspect on which I based was number of false positives. The side parameter was time of processing.

After some tuning I ended up with perticular parameters:

| Feature | Value |
|---------|-------|
| Color space | YCrCb |
| Orientations | 8 |
| Pixels per cell | 8x8 |
| Cells per block | 2x2 |
| Color channels  | ALL |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in `svc.py` file, main training function is between the `120` and `166` line.

I trained the LinearSVC with the hog features and histogram features. First I extracted the features from both datasets, then I mixed and scaled them to create trainig set. 20% of the training set was splitted to create test set.

After trainig the classifier was saved to file to avoid tarining on each test.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window implementation was done in `detector.py` file, lines `60-127`.

Scales switching was done in the same file (lines `132-135`).

I ended up with four scales: 1 and 1.5

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![][detection]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The heatmap threshold was settled to 2.

Here's an example result showing the heatmap:

![][heatmap]

### Here are four frames, their corresponding heatmaps, and final bounding box:

![][pipeline_example]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I haven't implemented any averaging of the windows, so they jump between the frames. It would be required to have it.

The second issue that could be handled is detection of the vehicles on the lanes that are separated. 

