import glob

import cv2
import numpy as np


class CameraCalibrator:

    def __init__(self, calibration_images_dir, chessboard_width=9, chessboard_height=6):
        self.calibration_images_dir = calibration_images_dir
        self.chessboard_height = chessboard_height
        self.chessboard_width = chessboard_width
        self.is_calibrated = False
        self.ret = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.image_points, self.object_points, self.img_size = self._get_calibration_parameters()

    def calibrate(self):
        """
        Calibrate the camera.
        :return: the camera matrix and distortion coefficients.
        """

        self.ret, camera_matrix, distortion_coefficients, self.rotation_vectors, self.translation_vectors = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.img_size,
            None,
            None)

        self.is_calibrated = True

        return camera_matrix, distortion_coefficients

    def draw_chessboard_corners_on_calibrated_images(self):
        # Prepare object points and image points
        all_object_points = []  # 3D points in real world space
        all_image_points = []  # 2D points in image plane

        object_points = self._get_prepared_object_points()

        calibrated_chessboard_images = []

        # Step through the list and search for chessboard corners
        for image_filename in self._calibration_image_filenames():

            # read in each image
            img = cv2.imread(image_filename)

            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_width, self.chessboard_height), None)

            # If corners are found then add object points and image points
            if ret is True:
                all_image_points.append(corners)
                all_object_points.append(object_points)

                # draw and display the corners
                calibrated = cv2.drawChessboardCorners(img,
                                                       (self.chessboard_width, self.chessboard_height),
                                                       corners,
                                                       ret)

                calibrated_chessboard_images.append((img, image_filename))

        return calibrated_chessboard_images

    def _get_prepared_object_points(self):
        dimensions = 3
        object_points = np.zeros((self.chessboard_height * self.chessboard_width, dimensions), np.float32)
        object_points[:, :2] = np.mgrid[0:self.chessboard_width, 0:self.chessboard_height].T.reshape(-1, 2)
        return object_points

    def _get_calibration_parameters(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points and image points
        object_points = []  # 3D points in real world space
        image_points = []  # 2D points in image plane

        prepared_object_points = self._get_prepared_object_points()

        # Step through the list and search for chessboard corners
        for image_filename in self._calibration_image_filenames():
            # read in each image
            img = cv2.imread(image_filename)

            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_width, self.chessboard_height), None)

            # If corners are found then add object points and image points
            if ret is True:
                object_points.append(prepared_object_points)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners)
                img_size = (gray.shape[1], gray.shape[0])

        return image_points, object_points, img_size

    def _calibration_image_filenames(self):
        # Make a list of calibration images where the filenames follow the pattern calibration*.jpg
        image_name_pattern = self.calibration_images_dir + '/calibration*.jpg'

        filenames = glob.glob(image_name_pattern)

        return filenames
