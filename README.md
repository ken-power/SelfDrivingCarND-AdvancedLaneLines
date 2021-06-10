# Advanced Lane Finding

This project is part of [Udacity](https://www.udacity.com)'s [Self-driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) program. The goal of this proejct is to write a software pipeline to identify the lane boundaries in a video. 


# The Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.




# Code structure
The code is in a module called `lane_finding` which is in the [lane_finding](lane_finding) directory. 

There are two files that run the `lane_finding` module:
* [process_images.py](process_images.py) executes the `lane_finding` module and runs the set of test images through the image processing pipeline.
* [process_video.py](process_video.py) executes the `lane_finding` module and runs the three test videos through the image processing pipeline.

## Pipeline summary

The camera calibration matrix and distortion coefficients are computed when the pipeline is initialized. 
```python
# Calibrate the camera when the pipeline is first created
self.camera_matrix, self.distortion_coefficients = self.camera_calibrator.calibrate()
```

The `process()` function in the [Pipeline](lane_finding/controller/pipeline.py) class is hte main controller for executing the pipeline.
```python
# STEP 1. Undistort the image using the coefficients found in camera calibration
undistorted_image = self.image_undistorter.undistort(image, self.camera_matrix, self.distortion_coefficients)

# STEP 2. Apply thresholds to create a binary image, getting the binary image to highlight the lane lines as
# much as possible
binary_image = self.threshold_converter.binary_image(undistorted_image, self.hyperparameters)

# STEP 3. Apply a perspective transform to obtain bird's eye view
birdseye_image, M, Minv = self.perspective_transformer.birdseye_transform(binary_image)

# STEP 4. Find the lane lines
self.left_fitx, self.right_fitx, self.ploty, birdseye_with_lane_lines = self.lane.find_lane_lines(
    birdseye_image,
    self.lane,
    self.hyperparameters.image_frame_number)

# STEP 5. Draw the lane lines area back onto the original image
image_with_detected_lane = self.image_builder.draw_lane_on_road(undistorted_image,
                                                                Minv,
                                                                self.lane.left_line,
                                                                self.lane.right_line,
                                                                keep_state=self.hyperparameters.keep_state)

```

```python
# STEP 6. Add inset images and text to the main image
self.calculate_lane_metrics(birdseye_image.shape[0], birdseye_image.shape[1])

# Add metrics and images from different pipeline steps as insets overlaid on the main image
final_image = self.image_builder.add_overlays_to_main_image(image_with_detected_lane,
                                                            binary_image,
                                                            birdseye_image,
                                                            birdseye_with_lane_lines,
                                                            undistorted_image,
                                                            self.radius_of_curvature_metres,
                                                            self.offset_in_meters,
                                                            str(self.offset_position),
                                                            self.hyperparameters.image_frame_number)
```

# Writeup

## Camera calibration
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

The images for camera calibration are stored in the folder called [camera_cal](data/camera_cal).  The images in [test_images](data/test_images) are for testing the pipeline on single frames.  

## Pipeline

### Distortion correction
* Apply a distortion correction to raw images.

### Thresholded binary image
* Use color transforms, gradients, etc., to create a thresholded binary image.

### Perspective transform
* Apply a perspective transform to rectify binary image ("birds-eye view").

### Identify lane pixels
* Detect lane pixels and fit to find the lane boundary.

### Radius of curvature
* Determine the curvature of the lane and vehicle position with respect to center.

### Plotting lane back onto the road
* Warp the detected lane boundaries back onto the original image.

## Pipeline video
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Project video

The input video is [here](data/test_videos/project_video.mp4).

The output video is [here](output_videos/out_project_video.mp4):

![project video output](output_videos/out_project_video.mp4)

### Optional challenge videos

The [challenge_video.mp4](data/test_videos/challenge_video.mp4) video is an extra (and optional) challenge to test the pipeline under somewhat trickier conditions.  The [harder_challenge.mp4](data/test_videos/harder_challenge_video.mp4) video is another optional challenge and is brutal!


#### Challenge video

The input video is [here](data/test_videos/challenge_video.mp4).

The output video is [here](output_videos/out_challenge_video.mp4):


#### Harder challenge video
The input video is [here](data/test_videos/harder_challenge_video.mp4).

The output video is [here](output_videos/out_harder_challenge_video.mp4):

## Discussion


