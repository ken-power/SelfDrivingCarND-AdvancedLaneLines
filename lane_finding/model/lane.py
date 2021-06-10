import cv2
import numpy as np
from matplotlib import pyplot as plt

from lane_finding.model.line import Line


class Lane:
    """
    Represents a road lane.
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

        self.left_line = Line(hyperparameters)
        self.right_line = Line(hyperparameters)

        self.left_fit_poly_coeffs = None
        self.right_fit_poly_coeffs = None

    def histogram(self, img):
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0] // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    def find_lane_lines(self,
                        birdseye_binary,
                        prev_lane,
                        image_frame_num=0):

        if image_frame_num == 0 and self.hyperparameters.reset_lane_search is True:
            return self.get_polynomial_coeffs_using_sliding_window(
                birdseye_binary,
                prev_lane)
        else:
            self.hyperparameters.reset_lane_search = False
            return self.get_polynomial_coeffs_using_previous_laneline_position(
                birdseye_binary,
                prev_lane)

    def get_polynomial_coeffs_using_sliding_window(self,
                                                   birdseye_binary,
                                                   prev_lane):
        """
        Get polynomial coefficients for lane-lines detected in an binary image.

        :param birdseye_binary: input bird's eye view binary image
        :param prev_left_lane_line: left lane-line previously detected
        :param prev_right_lane_line: left lane-line previously detected
        :return: updated lane line coefficients and output image
        """
        height, width = birdseye_binary.shape

        # Take a histogram of the bottom half of the image
        # histogram = np.sum(birdseye_binary[height // 2:-30, :], axis=0)
        histogram = self.histogram(birdseye_binary)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((birdseye_binary, birdseye_binary, birdseye_binary)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = int(height / self.hyperparameters.lane().num_sliding_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = birdseye_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = self.hyperparameters.lane().margin_first_frame  # width of the windows +/- margin
        minpix = self.hyperparameters.lane().minipix_first_frame  # minimum number of pixels found to recenter window

        # Create empty lists to receive left and right lane pixel indices
        left_lane_pixel_indices = []
        right_lane_pixel_indices = []

        # Step through the windows one by one
        for window in range(self.hyperparameters.lane().num_sliding_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,
                          (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high),
                          (0, 255, 0),
                          2)
            cv2.rectangle(out_img,
                          (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high),
                          (0, 255, 0),
                          2)

            # Identify the nonzero pixels in x and y within the window
            good_left_pixel_indices = \
                ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                 & (nonzero_x < win_xleft_high)).nonzero()[0]

            good_right_pixel_indices = \
                ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                 & (nonzero_x < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_pixel_indices.append(good_left_pixel_indices)
            right_lane_pixel_indices.append(good_right_pixel_indices)

            # If we found > minpix pixels, recenter next window on their mean position
            if len(good_left_pixel_indices) > minpix:
                leftx_current = int(np.mean(nonzero_x[good_left_pixel_indices]))
            if len(good_right_pixel_indices) > minpix:
                rightx_current = int(np.mean(nonzero_x[good_right_pixel_indices]))

        # Concatenate the arrays of indices
        try:
            left_lane_pixel_indices = np.concatenate(left_lane_pixel_indices)
            right_lane_pixel_indices = np.concatenate(right_lane_pixel_indices)
        except ValueError:
            self.left_line.reset()
            self.right_line.reset()
            pass

        # Extract left and right line pixel positions
        left_line_pixel_positions_x = nonzero_x[left_lane_pixel_indices]
        left_line_pixel_positions_y = nonzero_y[left_lane_pixel_indices]
        right_line_pixel_positions_x = nonzero_x[right_lane_pixel_indices]
        right_line_pixel_positions_y = nonzero_y[right_lane_pixel_indices]

        # Resetting so np.polyfit does not fail
        if len(left_line_pixel_positions_x) == 0: left_line_pixel_positions_x = [1, 1, 1]
        if len(left_line_pixel_positions_y) == 0: left_line_pixel_positions_y = [1, 1, 1]
        if len(right_line_pixel_positions_x) == 0: right_line_pixel_positions_x = [1, 1, 1]
        if len(right_line_pixel_positions_y) == 0: right_line_pixel_positions_y = [1, 1, 1]

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(left_line_pixel_positions_y, left_line_pixel_positions_x, 2)
        right_fit = np.polyfit(right_line_pixel_positions_y, right_line_pixel_positions_x, 2)

        prev_lane.left_line.update(left_fit)
        prev_lane.right_line.update(right_fit)

        self.left_fit_poly_coeffs = left_fit
        self.right_fit_poly_coeffs = right_fit

        # Generate x and y values for plotting
        plot_y = np.linspace(0, birdseye_binary.shape[0] - 1, birdseye_binary.shape[0])
        try:
            left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
            right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        except TypeError:
            # Avoids an error if `left_fit` and `right_fit` are still none or incorrect
            print('Error - resetting lines: The function failed to fit a line')
            self._reset_and_go_back_to_sliding_window()

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[left_line_pixel_positions_y, left_line_pixel_positions_x] = [255, 0, 0]
        out_img[right_line_pixel_positions_y, right_line_pixel_positions_x] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, plot_y, color='yellow')
        plt.plot(right_fitx, plot_y, color='yellow')

        return left_fitx, right_fitx, plot_y, out_img

    def get_polynomial_coeffs_using_previous_laneline_position(self,
                                                               binary_warped,
                                                               prev_lane):
        """
        In the second and subsequent frames of video we don't need to do a blind search again, but instead we can just
        search in a margin around the previous lane line position, like in the above image. The green shaded area shows
        where we searched for the lines this time. So, once we know where the lines are in one frame of video, we can
        do a highly targeted search for them in the next frame.

        This is equivalent to using a customized region of interest for each frame of video, and should help us track
        the lanes through sharp curves and tricky conditions. If we lose track of the lines, we go back to our sliding
        windows search to rediscover them.

        :param binary_warped: a binary warped image
        :param prev_lane: previous lane, containing the polynomial function for the left and right lane lines
        :return: the polynomial coefficients for the lane lines of the current image frame
        """
        prev_lane_copy = prev_lane
        # The width of the margin around the previous polynomial to search
        margin = self.hyperparameters.lane().margin_second_frame

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = self.left_fit_poly_coeffs
        right_fit = self.right_fit_poly_coeffs

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_pixel_indices = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                                left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                      left_fit[1] * nonzeroy + left_fit[
                                                                                          2] + margin)))

        right_lane_pixel_indices = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                                 right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                        right_fit[1] * nonzeroy +
                                                                                        right_fit[
                                                                                            2] + margin)))

        # Again, extract left and right line pixel positions
        left_line_pixel_positions_x = nonzerox[left_lane_pixel_indices]
        left_line_pixel_positions_y = nonzeroy[left_lane_pixel_indices]
        right_line_pixel_positions_x = nonzerox[right_lane_pixel_indices]
        right_line_pixel_positions_y = nonzeroy[right_lane_pixel_indices]

        try:
            if (right_line_pixel_positions_x[0] - left_line_pixel_positions_x[
                0] < self.hyperparameters.lane().min_lane_projection_width) or \
                    (right_line_pixel_positions_x[0] - left_line_pixel_positions_x[
                        0] > self.hyperparameters.lane().max_lane_projection_width):
                return self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)
        except:
            print("EX: index 0 is out of bounds at frame", self.hyperparameters.image_frame_number)
            return self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)
            pass

        # Fit new polynomials:
        # Fit a second order polynomial to each line with np.polyfit()
        try:
            left_fit = np.polyfit(left_line_pixel_positions_y, left_line_pixel_positions_x, 2)
            right_fit = np.polyfit(right_line_pixel_positions_y, right_line_pixel_positions_x, 2)
        except:
            # print("Exception: left_fit={} right_fit={}".format(left_fit, right_fit))
            self.left_line.reset()
            self.right_line.reset()

        # Generate x and y values for plotting
        plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        if self.hyperparameters.thresholding().thresholding_function() == 3:
            # print("DEBUG (search around poly): changing the projected lane distance")
            left_fit = left_fit[:len(left_fit) // 2]
            right_fit = right_fit[:len(right_fit) // 2]
            plot_y = plot_y[:len(plot_y) // 2]

            for i in range(len(left_fit)):
                if left_fit[i] >= right_fit[i]:
                    np.delete(left_fit, i)
                    np.delete(right_fit, i)
                    # self.reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)
                    # break

        if len(left_fit) < 3:  # We don't have enough polynomial coefficients
            return self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

        if right_fit[2] - left_fit[2] > self.hyperparameters.lane().poly_fit_val:
            # Calculate both polynomials using plot_y, left_fit and right_fit
            left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
            right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

            self._handle_difficult_road_conditions(binary_warped, left_fitx, prev_lane, prev_lane_copy, right_fitx)

            prev_lane.left_line.update(left_fit, is_detected=True)
            prev_lane.right_line.update(right_fit, is_detected=True)
            self.left_fit_poly_coeffs = left_fit
            self.right_fit_poly_coeffs = right_fit

            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)

            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_pixel_indices], nonzerox[left_lane_pixel_indices]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_pixel_indices], nonzerox[right_lane_pixel_indices]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, plot_y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                            plot_y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, plot_y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                             plot_y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, plot_y, color='yellow')
            plt.plot(right_fitx, plot_y, color='yellow')
            ## End visualization steps ##

            return left_fitx, right_fitx, plot_y, out_img

        else:
            # If we're in here it's because we encountered a problem finding the lanes
            # print("Frame {}: Compensating and resetting".format(self.hyperparameters.image_frame_number))
            return self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

    def offset_and_position(self,
                            img_h,
                            img_w):
        """
        Compute offset from center of the inferred lane. The offset from the lane center can be computed under the
        hypothesis that the camera is fixed and mounted in the midpoint of the car roof. In this case, we can
        approximate the car's deviation from the lane center as the distance between the center of the image and the
        midpoint at the bottom of the image of the two lane-lines detected.

        :param img_h: the height of the birdseye image
        :param img_w: the width of the birdseye image
        :return: offset ond position of the vehicle, relative to the center of the lane
        """
        # Vehicle position with respect to camera mounted at the center of the car
        vehicle_position = img_w / 2

        left_fit = self.left_line.current_fit_coeffs
        right_fit = self.right_line.current_fit_coeffs

        # Calculate x-intercept for the left and right polynomial
        left_fit_x_int = left_fit[0] * img_h ** 2 + left_fit[1] * img_h + left_fit[2]
        right_fit_x_int = right_fit[0] * img_h ** 2 + right_fit[1] * img_h + right_fit[2]

        # Calculate lane center position from x-intercepts
        lane_center_position = (left_fit_x_int + right_fit_x_int) / 2

        offset = np.abs(vehicle_position - lane_center_position) * self.hyperparameters.lane().metres_per_pixel_x

        # Check if vehicle's position is left or right of center of the lane
        if lane_center_position == vehicle_position:
            position = "center"
        elif lane_center_position > vehicle_position:
            position = "left"
        else:
            position = "right"

        return offset, position

    def radius(self, left_fit_cr, right_fit_cr, plot_y):
        """
        Calculate the radius of curvature for the lane, based on calculating the radius for each of it's lines.
        """
        left_line_radius = self.left_line.radius_of_curvature(left_fit_cr, plot_y)
        right_line_radius = self.right_line.radius_of_curvature(right_fit_cr, plot_y)

        return np.average([left_line_radius, right_line_radius])

    def _handle_difficult_road_conditions(self, binary_warped, left_fitx, prev_lane, prev_lane_copy, right_fitx):
        lane_center_position = (left_fitx + right_fitx) // 2
        center_threshold = 100
        min_vehicle_width = 400
        image_center_x = 600
        pixel_num = 20

        if left_fitx[pixel_num] > (lane_center_position[pixel_num] - center_threshold):
            print("\t - Left line crossed center threshold")
            self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

        if right_fitx[pixel_num] < (lane_center_position[pixel_num] - center_threshold):
            print("\t - Right line crossed center threshold")
            self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

        if left_fitx[pixel_num] >= image_center_x:
            print("\t - Left line crossed center of image")
            self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

        if right_fitx[pixel_num] <= image_center_x:
            print("\t - Right line crossed center of image")
            self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

        if right_fitx[pixel_num] - left_fitx[pixel_num] < min_vehicle_width:
            print("\t - (Right line x) - (Left line x) not wide enough for a vehicle")
            self._reset_and_go_back_to_sliding_window(binary_warped, prev_lane, prev_lane_copy)

    def _reset_and_go_back_to_sliding_window(self, binary_warped, prev_lane, prev_lane_copy):
        self.hyperparameters.reset_lane_search = True
        prev_lane.left_line.reset()
        prev_lane.right_line.reset()
        return self.get_polynomial_coeffs_using_sliding_window(binary_warped,
                                                               prev_lane_copy)
