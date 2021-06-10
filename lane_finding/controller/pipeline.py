import matplotlib.pyplot as plt
import numpy as np

from lane_finding.config.config_data import PATH_ROOT
from lane_finding.model.camera_calibrator import CameraCalibrator
from lane_finding.model.color_threshold_converter import ColorThresholdConverter
from lane_finding.model.distortion_corrector import DistortionCorrector
from lane_finding.model.lane import Lane
from lane_finding.model.perspective_transformer import PerspectiveTransformer
from lane_finding.view.image_builder import ImageBuilder
from lane_finding.view.image_plotter import ImagePlotter


class Pipeline:
    def __init__(self, calibration_images_dir, hyperparameters, debug_pipeline_images=False):
        """
        Initialise the image processing pipeline.

        :param calibration_images_dir: the directory containing the camera calibration images
        :param hyperparameters: hyperparameters used in processing images through the pipeline
        :param debug_pipeline_images: set to True to output pipeline images for each frame
        """
        self.hyperparameters = hyperparameters

        self.CALIBRATION_IMAGES_DIR = calibration_images_dir

        self.camera_calibrator = CameraCalibrator(self.CALIBRATION_IMAGES_DIR)
        self.distortion_corrector = DistortionCorrector()
        self.perspective_transformer = PerspectiveTransformer()
        self.threshold_converter = ColorThresholdConverter()
        self.lane = Lane(self.hyperparameters)

        # Calibrate the camera when the pipeline is first created
        self.camera_matrix, self.distortion_coefficients = self.camera_calibrator.calibrate()

        self.radius_of_curvature_metres = 0
        self.offset_in_meters = 0
        self.offset_position = 'center'

        self.plotter = ImagePlotter()
        self.image_builder = ImageBuilder()

        # Debug options
        self.debug_pipeline_images = debug_pipeline_images
        self.save_pipeline_images_to_disk = True

    def process(self, image, add_insets=True):
        """
        Pass an input color image through the full image processing pipeline. The camera has already been calibrated
        as part of initializing the pipeline.

        :param image: input color image
        :param add_insets: if True, add metrics and images from pipeline steps as insets overlaid on the main image
        :return: combined output image with detected lane overlaid on the road surface
        """
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

        # STEP 6. Add inset images and text to the main image
        if add_insets:
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
        else:
            final_image = image_with_detected_lane

        self.handle_pipeline_images(image,
                                    undistorted_image,
                                    binary_image,
                                    birdseye_image,
                                    birdseye_with_lane_lines,
                                    image_with_detected_lane,
                                    final_image)

        self.hyperparameters.image_frame_number += 1

        return final_image

    def calculate_lane_metrics(self, img_h, img_w):

        # Calculate the radius of curvature (in meters)
        left_fit_cr = np.polyfit(self.ploty * self.hyperparameters.lane().metres_per_pixel_y,
                                 self.left_fitx * self.hyperparameters.lane().metres_per_pixel_x,
                                 2)

        right_fit_cr = np.polyfit(self.ploty * self.hyperparameters.lane().metres_per_pixel_y,
                                  self.right_fitx * self.hyperparameters.lane().metres_per_pixel_x,
                                  2)

        self.radius_of_curvature_metres = self.lane.radius(left_fit_cr, right_fit_cr, self.ploty)

        # Calculate the offset from center of the lane (in meters)
        self.offset_in_meters, self.offset_position = self.lane.offset_and_position(img_h, img_w)

    def handle_pipeline_images(self, original_image, undistorted_image, binary_image, birdseye_image,
                               birdseye_highlighted_lanes,
                               lane_area_on_road_img, final_image):
        if self.debug_pipeline_images:
            self.plotter.plot_5_images_side_by_side_cmap(
                undistorted_image, "Frame #{}: undistorted".format(self.hyperparameters.image_frame_number),
                binary_image, "binary",
                birdseye_image, "birdseye transform",
                birdseye_highlighted_lanes, "found lanes",
                lane_area_on_road_img, "lane area")
            plt.show()

        if self.save_pipeline_images_to_disk:
            self.save_frames(original_image, undistorted_image, binary_image, birdseye_image,
                             birdseye_highlighted_lanes, lane_area_on_road_img, final_image)

    def save_frames(self, original_image, undistorted_image, binary_image, birdseye_image, birdseye_highlighted_lanes,
                    lane_area_on_road_img, final_image):
        base_dir = PATH_ROOT + "data/test_pipeline_images/images_from_project_video/"
        file_ext = ".jpg"
        frames_to_save = [20, 111, 314, 553, 607, 740, 949, 989, 1016, 1036, 1118, 1260]
        if self.hyperparameters.image_frame_number in frames_to_save:
            plt.imsave(base_dir + "{}_0_original{}".format(self.hyperparameters.image_frame_number, file_ext),
                       original_image)
            plt.imsave(base_dir + "{}_1_undistorted{}".format(self.hyperparameters.image_frame_number, file_ext),
                       undistorted_image)
            plt.imsave(base_dir + "{}_2_binary{}".format(self.hyperparameters.image_frame_number, file_ext),
                       binary_image,
                       cmap='gray')
            plt.imsave(base_dir + "{}_3_birdseye{}".format(self.hyperparameters.image_frame_number, file_ext),
                       birdseye_image,
                       cmap='gray')
            plt.imsave(base_dir + "{}_4_birdseye_lanes{}".format(self.hyperparameters.image_frame_number, file_ext),
                       birdseye_highlighted_lanes)
            plt.imsave(
                base_dir + "{}_5_highlighted_area{}".format(self.hyperparameters.image_frame_number, file_ext),
                lane_area_on_road_img)
            plt.imsave(
                base_dir + "{}_6_final_image{}".format(self.hyperparameters.image_frame_number, file_ext),
                final_image)
