import os

import cv2
import matplotlib.pyplot as plt

from lane_finding.controller.pipeline import Pipeline

TEST_IAMGES_DIR = '../data/test_images'
CALIBRATION_IMAGES_DIR = '../data/camera_cal'

if __name__ == '__main__':

    test_files = []
    pipeline = Pipeline(CALIBRATION_IMAGES_DIR)

    for filename in os.listdir(TEST_IAMGES_DIR):
        if filename.endswith(".jpg"):
            filepath = os.path.join(TEST_IAMGES_DIR, filename)
            test_files.append(filepath)

    for test_img in test_files:
        frame = cv2.imread(test_img)

        final_image = pipeline.process(frame, keep_state=False)

        cv2.imwrite('output_images/out_{}'.format(os.path.basename(test_img)), final_image)

        plt.imshow(cv2.cvtColor(final_image, code=cv2.COLOR_BGR2RGB))
        plt.show()
