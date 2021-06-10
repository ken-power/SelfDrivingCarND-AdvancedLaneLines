CALIBRATION_IMAGES_DIR = 'data/camera_cal'

from moviepy.editor import VideoFileClip

from controller.hyperparameters import Hyperparameters
from controller.pipeline import Pipeline
from test_data import CALIBRATION_IMAGES_DIR
from test_data import TEST_VIDEOS_DIR
from test_data import VIDEO_OUTPUT_FILE_BASE_PATH


class PipelineVideoTests:
    def __init__(self):
        self.debug_images = False
        self.enable_audio = False

    def test_project_video(self):
        params = Hyperparameters()

        pipeline = Pipeline(CALIBRATION_IMAGES_DIR, params, debug_pipeline_images=self.debug_images)
        video_name = 'project_video.mp4'
        video_file_path = TEST_VIDEOS_DIR + '/' + video_name

        clip = VideoFileClip(video_file_path).fl_image(pipeline.process)
        clip.write_videofile(VIDEO_OUTPUT_FILE_BASE_PATH + video_name, audio=self.enable_audio)

    def test_challenge_video(self):
        params = Hyperparameters()
        params.lane().set_meters_per_pixel_y(20, 700)
        params.lane().set_meters_per_pixel_x(3.2, 675)
        params.lane().set_margin_first_frame(100)
        params.lane().set_margin_second_frame(80)
        params.lane().set_num_windows(35)
        params.lane().set_lane_projection_width(350, 710)
        params.lane().set_poly_fit_val(300)
        params.thresholding().set_sobel_threshold(130, 250)
        params.thresholding().set_sobelx_threshold(10, 150)
        params.thresholding().set_sobel_kernel_size(5)

        pipeline = Pipeline(CALIBRATION_IMAGES_DIR, params, debug_pipeline_images=self.debug_images)
        video_name = 'challenge_video.mp4'
        video_file_path = TEST_VIDEOS_DIR + '/' + video_name

        clip = VideoFileClip(video_file_path).fl_image(pipeline.process)
        clip.write_videofile(VIDEO_OUTPUT_FILE_BASE_PATH + video_name, audio=self.enable_audio)

    def test_harder_challenge_video(self):
        params = Hyperparameters()
        params.lane().set_meters_per_pixel_y(4, 420)
        params.lane().set_meters_per_pixel_x(1.2, 400)
        params.lane().set_margin_first_frame(90)
        params.lane().set_margin_second_frame(75)
        params.lane().set_num_windows(50)
        params.lane().set_minipix_first_frame(20)
        params.lane().set_minipix_second_frame(25)
        params.lane().set_lane_projection_width(350, 680)
        params.lane().set_poly_fit_val(200)
        params.thresholding().set_sobel_threshold(130, 220)
        params.thresholding().set_sobelx_threshold(25, 201)
        params.thresholding().set_sobel_kernel_size(3)
        params.thresholding().set_thresholding_function(3)

        pipeline = Pipeline(CALIBRATION_IMAGES_DIR, params, debug_pipeline_images=self.debug_images)
        video_name = 'harder_challenge_video.mp4'
        video_file_path = TEST_VIDEOS_DIR + '/' + video_name

        clip = VideoFileClip(video_file_path).fl_image(pipeline.process)
        clip.write_videofile(VIDEO_OUTPUT_FILE_BASE_PATH + video_name, audio=self.enable_audio)


if __name__ == '__main__':
    video_tests = PipelineVideoTests()

    video_tests.test_project_video()
    video_tests.test_challenge_video()
    video_tests.test_harder_challenge_video()
