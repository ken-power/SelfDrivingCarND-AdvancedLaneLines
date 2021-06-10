import cv2


class ImageUndistorter:
    """
    Encapsulate the process of undistorting images.
    """

    def undistort(self, image, camera_matrix, distortion_coefficients):
        """
        Perform image distortion correction and return the undistorted image.

        :param image: the original image captured by the camera
        :param camera_matrix: the camera matrix created when calibrating the camera
        :param distortion_coefficients: the distortion coefficients created when calibrating the camera

        :return an undistorted version of the input image
        """
        undistorted_image = cv2.undistort(image,
                                          camera_matrix,
                                          distortion_coefficients,
                                          None,
                                          newCameraMatrix=camera_matrix)

        return undistorted_image
