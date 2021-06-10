from moviepy.editor import VideoFileClip

from controller.pipeline import Pipeline
from controller.hyperparameters import Hyperparameters

CALIBRATION_IMAGES_DIR = 'data/camera_cal'

if __name__ == '__main__':
    hyper_params = Hyperparameters()

    pipeline = Pipeline(CALIBRATION_IMAGES_DIR, hyper_params)

    clip = VideoFileClip('data/test_videos/project_video.mp4').fl_image(pipeline.process)
    clip.write_videofile('output_videos/out_project_video.mp4', audio=False)
