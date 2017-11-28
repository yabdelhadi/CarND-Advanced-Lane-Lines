## **Advanced Lane Finding Project**

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.JPG "Undistorted"
[image2]: ./examples/img_undistort_output.JPG "Undistorted"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/skip_color_fit_lines.JPG "Fit Visual"
[image7]: ./examples/output_straight_lines1.jpg "Output"
[image8]: ./examples/output_test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in cell 2 of the IPython notebook "Advanced_lane_lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I used the image below to correct the distortion. The distortion corrected image is shown below which shows the difference between the original and undistorted image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are contained in contained in cell 2 & 5 of the IPython notebook "Advanced_lane_lines.ipynb". I experiment with different colorspaces to overcome some of the challenges in this project like detected yellow lines and detecting lines under different lighting conditions and shadow. I was able to achieve the best results by using the HSV and HLS colorspaces (S and V channels specifically) and tuning their thresholds accordingly. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which contained in cell 5 of the IPython notebook "Advanced_lane_lines.ipynb". The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used a function called `sliding_window()` which take a histogram of a thresholded warped image. Using this histogram, I then find the peaks to identify lane lines. The `sliding_window()` function uses a sliding window to detect a certian number of pixels within each window. Those pixels are then used to fit them using a second order polynomial. The code for this step is contained in cell 7 of the IPython notebook "Advanced_lane_lines.ipynb". An example of the output is shown below

![alt text][image5]

Now that I know where the lane lines are located, I don't have to search from scratch again so I use a function called `skip_search()` which searches for the lane line pixels in certain areas based on the last lane detection. The code for this step is contained in cell 7 of the IPython notebook "Advanced_lane_lines.ipynb". An example of the output is shown below

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used a function called `curvature_radius()` which calculate the radius of curvature of the left and right lane line then I used the mean to calculate the lane radius of curvature. The code for this step is contained in cell 7 of the IPython notebook "Advanced_lane_lines.ipynb". 

I then used a function called `lane_offset()` which assumes that the camera is mounted on the center of the vehicle windshield so it calculates the camera position and then calculates the center of the lane using the polynomial fit x-values. Now I know where the camera is located and where the center of the lane, I can calculate the offset by subtracting them. The code for this step is contained in cell 7 of the IPython notebook "Advanced_lane_lines.ipynb" and it is also shown below
```python
def lane_offset(leftx, rightx, img):
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    camera_position = img.shape[1]/2
    lane_center = (rightx[-1] + leftx[-1])/2
    center_offset_pixels = abs(camera_position - lane_center)
    offset = center_offset_pixels * xm_per_pix
    
    return offset
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 11 & 12 in the function `process_images()`.  Here is an example of my result on a test image:

![alt text][image7]

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a lot of challenges detecting the lane markers correctly especially in the lighter pavement areas. I tried a lot of different colorspaces with different threshold. I ended up using the HSV and HLS colorspaces since it seems to be the best with the right threshold. Tuning the threshold was very challenging but fun. There was also some area where the shadows and skid marks on the road caused my pipeline to detect the lane a lot wider then it is actually is so I decided to implement a function called `region_of_interest()` which masks the region of interest

There is two frames at the end of the lighter pavement area where the lines wobble a little bit which won't cause a catastrophic failure or let the car to drive off the road. I can probably make my pipeline more robust by dropping bad frames, taking the average of N frames, or specify the lane width with +/- a threshold. 

Overall I think my pipeline does a good job in detected the lane line correctly.
