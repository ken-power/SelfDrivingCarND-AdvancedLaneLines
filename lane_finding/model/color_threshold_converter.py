import cv2
import numpy as np


class ColorThresholdConverter:

    def binary_image(self, img, hyperparameters):
        """
        Serve as a facade so I can select different pipelines

        :param img: undistorted color image
        :return: binary image
        """
        sxbinary, s_binary, color_binary, combined_binary = self.convert_to_binary(img, hyperparameters)

        # The current implementation of the main pipeline just needs one binary image
        return combined_binary

    def convert_to_binary(self, img, hyperparameters):
        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = self.HLS().l_channel(hls)
        s_channel = self.HLS().s_channel(hls)

        # Sobel x
        kernel_size = hyperparameters.thresholding().sobel_kernel_size()
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = hyperparameters.thresholding().sobelx_threshold()[0]
        thresh_max = hyperparameters.thresholding().sobelx_threshold()[1]
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = hyperparameters.thresholding().sobel_threshold()[0]
        s_thresh_max = hyperparameters.thresholding().sobel_threshold()[1]
        s_binary = self.HLS().s_binary(s_channel, thresh=(s_thresh_min, s_thresh_max))

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return sxbinary, s_binary, color_binary, combined_binary

    def hls_select(self, img, thresh=(0, 255)):
        '''
        Thresholds the S-channel of HLS
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

        return binary_output

    def _thresh_frame_in_hsv(self, image, min_values, max_values):
        """
        Threshold a color image in HSV space
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_th_ok = np.all(hsv > min_values, axis=2)
        max_th_ok = np.all(hsv < max_values, axis=2)

        out = np.logical_and(min_th_ok, max_th_ok)

        return out

    def _thresh_frame_sobel(self, image, kernel_size):
        """
        Apply Sobel edge detection to an input image, then threshold the result
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

        _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

        return sobel_mag.astype(bool)

    def _get_binary_from_equalized_grayscale(self, image):
        """
        Apply histogram equalization to an input image, threshold it and return the (binary) result.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        eq_global = cv2.equalizeHist(gray)

        _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

        return th

    def combined_threshold(self, img):
        abs_bin = self.abs_sobel_threshold(img, orient='x', thresh=(50, 255))

        mag_bin = self.mag_threshold(img, sobel_kernel=3, mag_thresh=(50, 255))
        dir_bin = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

        combined = np.zeros_like(dir_bin)
        combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1)))] = 1

        return combined

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_threshold(self, img, orient='x', sobel_kernel=(3, 3), thresh=(0, 255)):

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Remove noise
        gray = cv2.GaussianBlur(gray, sobel_kernel, 0)

        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)

        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function to threshold an image for a given range and Sobel kernel
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the gradient direction,
        # apply a threshold
        gradient_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        # create a binary image result
        binary_output = np.zeros_like(gradient_dir)
        binary_output[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def convert_gray_to_binary(self, gray, thresh):
        binary = np.zeros_like(gray)
        binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

        return binary

    class RGB:
        def r_channel(self, image):
            return image[:, :, 0]

        def g_channel(self, image):
            return image[:, :, 1]

        def b_channel(self, image):
            return image[:, :, 2]

        def r_binary(self, R, thresh):
            binary = np.zeros_like(R)
            binary[(R > thresh[0]) & (R <= thresh[1])] = 1

            return binary

    class HLS:
        def h_channel(self, image):
            return image[:, :, 0]

        def l_channel(self, image):
            return image[:, :, 1]

        def s_channel(self, image):
            return image[:, :, 2]

        def s_binary(self, S, thresh):
            binary = np.zeros_like(S)
            binary[(S >= thresh[0]) & (S <= thresh[1])] = 1

            return binary

        def h_binary(self, H, thresh):
            binary = np.zeros_like(H)
            binary[(H > thresh[0]) & (H <= thresh[1])] = 1

            return binary
