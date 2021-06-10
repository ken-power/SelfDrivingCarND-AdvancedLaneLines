# Advanced Lane Finding

This project is part of [Udacity](https://www.udacity.com)'s [Self-driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) program. The goal of this proejct is to write a software pipeline to identify the lane boundaries in a video. 


# Project goals

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.




# Solution overview
The code is in a module called `lane_finding` which is in the [lane_finding](lane_finding) directory. 

There are two files that execute the `lane_finding` module:
* [process_images.py](process_images.py) executes the `lane_finding` module and runs the set of test images through the image processing pipeline.
* [process_video.py](process_video.py) executes the `lane_finding` module and runs the three test videos through the image processing pipeline.

## Code structure
The design of the lane finding module follows a basic Model-View-Controller pattern. This directory layout reflects this pattern, with each part in its own sub-directory:
```text
lane_finding/
  |
  |--model/
  |--view/
  |--controller/
```

* [Model](lane_finding/model) contains the core abstractions for camera calibration ([CameraCalibrator](lane_finding/model/camera_calibrator.py)), distortion correction ([DistortionCorrector](lane_finding/model/distortion_corrector.py)), color thresholding ([ColorThresholdConverter](lane_finding/model/color_threshold_converter.py)), perspective transformation([PerspectiveTransformer](lane_finding/model/perspective_transformer.py)), lanes ([Lane](lane_finding/model/lane.py)), and lane lines ([Line](lane_finding/model/line.py)).
* [View](lane_finding/view) contains two classes that manage the presentation logic. [ImagePlotter](lane_finding/view/image_plotter.py) takes care of plotting images, and provides a number of utility functions for plotting various image combinations. [ImageBuilder](lane_finding/view/image_builder.py) is that class that builds the final output image with the lane highlihgted on the road. This class manages the overlay of inset images and text on the main image. 
* [Controller](lane_finding/controller) contains the [Pipeline](lane_finding/controller/pipeline.py) class and a [HyperParameters](lane_finding/controller/hyperparameters.py) class. [Pipeline](lane_finding/controller/pipeline.py) processes images through the pipeline.  [HyperParameters](lane_finding/controller/hyperparameters.py) contains parameters that allow us to tune the image processing pipeline for different scenarios. For example, I use the same pipeline code on all three videos for this project, but tune the hyperparameters differently for each video. Examples [are shown below](#Project-video).


## Pipeline summary

The camera calibration matrix and distortion coefficients are computed when the pipeline is initialized. 
```python
# Calibrate the camera when the pipeline is first created
self.camera_matrix, self.distortion_coefficients = self.camera_calibrator.calibrate()
```

The `process()` function in the [Pipeline](lane_finding/controller/pipeline.py) class is the main controller for executing the pipeline.

```python
# STEP 1. Undistort the image using the coefficients found in camera calibration
undistorted_image = self.distortion_corrector.undistort(image, self.camera_matrix, self.distortion_coefficients)

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

Udacity provided two sets of images for testing. The folder called [camera_cal](data/camera_cal) contains the images for camera calibration.  The images in [test_images](data/test_images) are for testing the pipeline on single frames.  

In addition to these I created my own set of images for testing by saving image frames of the test videos.
* The [data/test_pipeline_images](data/test_pipeline_images) contains the images I generated for testing. 
* For each frame, there is an undistorted image, binary image, birdseye transform image, birdseye transform with highlihgted lanes, and an image that shows the lane projected onto the road surface.
* The [data/test_pipeline_images/images_from_project_video](data/test_pipeline_images/images_from_project_video) contains a set of images captured from specific frames of my output video. In addition to images for all of the intermediate pipeline steps, there are images for the original input frame and the finale combined result image frame.  

## Pipeline

### Distortion correction
* Apply a distortion correction to raw images.

Original Image | Undistorted Image
--- | ---
![Frame 20 Original](data/test_pipeline_images/images_from_project_video/20_0_original.jpg) | ![Frame 20 Undistorted](data/test_pipeline_images/images_from_project_video/20_1_undistorted.jpg) 

### Thresholded binary image
* Use color transforms, gradients, etc., to create a thresholded binary image.

Undistorted Image | Thresholded Binary Image
--- | ---
![Frame 20 Undistorted](data/test_pipeline_images/images_from_project_video/20_1_undistorted.jpg) | ![Frame 20 Binary](data/test_pipeline_images/images_from_project_video/20_2_binary.jpg) 

### Perspective transform
* Apply a perspective transform to rectify binary image ("birds-eye view").

Undistorted Image | Perspective Transform
--- | ---
![Frame 20 Undistorted](data/test_pipeline_images/images_from_project_video/20_1_undistorted.jpg) | ![Frame 20 Birdseye](data/test_pipeline_images/images_from_project_video/20_3_birdseye.jpg) 

### Identify lane pixels
* Detect lane pixels and fit to find the lane boundary.

Undistorted Image | Lane Boundary
--- | ---
![Frame 20 Undistorted](data/test_pipeline_images/images_from_project_video/20_1_undistorted.jpg) | ![Frame 20 Birdseye](data/test_pipeline_images/images_from_project_video/20_4_birdseye_lanes.jpg) 

### Radius of curvature
* Determine the curvature of the lane and vehicle position with respect to center.

### Plotting lane back onto the road
* Warp the detected lane boundaries back onto the original image.

Undistorted Image | Lane on Road Image
--- | ---
![Frame 20 Undistorted](data/test_pipeline_images/images_from_project_video/20_1_undistorted.jpg) | ![Frame 20 Birdseye](data/test_pipeline_images/images_from_project_video/20_5_highlighted_area.jpg) 

### Final output image

![Final Output Image Frmae 20](data/test_pipeline_images/images_from_project_video/20_6_final_image.jpg)

Here are some more examples of final image frames from my project video output. I chose these to show specific interesting points during the video. 

#### Frame 111
**Scenario**: Long stretch of relatively straight road with a bend up ahead to the left.

![Final Output Image Frmae 111](data/test_pipeline_images/images_from_project_video/111_6_final_image.jpg)

#### Frame 314
**Scenario**: Long stretch of relatively straight road with a car passing on the right.

![Final Output Image Frmae 314](data/test_pipeline_images/images_from_project_video/314_6_final_image.jpg)

#### Frame 553
**Scenario**: Passing through an area of road with a lot of bright light causing glare on the road, and making it harder to see the lane lines.

![Final Output Image Frmae 553](data/test_pipeline_images/images_from_project_video/553_6_final_image.jpg)

#### Frame 607
**Scenario**: Transitioning from an area of road with a lot of bright light causing glare on the road, back to more favorable lighting conditions.

![Final Output Image Frmae 607](data/test_pipeline_images/images_from_project_video/607_6_final_image.jpg)


## Pipeline video
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Project video

The input video is [here](data/test_videos/project_video.mp4). The output video is [here](output_videos/out_project_video.mp4):

You can watch the output of the pipeline applied to the project video on YouTube:
[![Output of project video](https://img.youtube.com/vi/kzYbIra3nH8/0.jpg)](https://youtu.be/kzYbIra3nH8 "Project video")


### Optional challenge videos

The [challenge_video.mp4](data/test_videos/challenge_video.mp4) video is an extra (and optional) challenge to test the pipeline under somewhat trickier conditions.  The [harder_challenge.mp4](data/test_videos/harder_challenge_video.mp4) video is another optional challenge and is brutal!


#### Challenge video

The input video is [here](data/test_videos/challenge_video.mp4). The output video is [here](output_videos/out_challenge_video.mp4):

You can watch the output of the pipeline applied to the challenge video on YouTube:
[![Challenge output](https://img.youtube.com/vi/M7CPvri28hE/0.jpg)](https://youtu.be/M7CPvri28hE "Challenge video")


#### Harder challenge video
The input video is [here](data/test_videos/harder_challenge_video.mp4). The output video is [here](output_videos/out_harder_challenge_video.mp4):

You can watch the output of the pipeline applied to the harder challenge video on YouTube:
[![Harder challenge output](https://img.youtube.com/vi/Q1qdfA6N8Iw/0.jpg)](https://youtu.be/Q1qdfA6N8Iw "Harder challenge video")

## Discussion


