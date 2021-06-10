import collections

import cv2
import numpy as np


class Line:
    """
    Represents a lane line on a road surface.
    """

    def __init__(self, hyperparameters, buffer_len=10):
        self.hyperparameters = hyperparameters

        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients averaged over the last n iterations
        self.best_fit_coeffs = collections.deque(maxlen=2 * buffer_len)

        # x values of the last n fits of the line
        self.recent_xfitted = [1, 1, 1]

        # average x values of the fitted line over the last n iterations
        self.bestx = [1, 1, 1]

        # polynomial coefficients for the most recent fit
        self.current_fit_coeffs = [np.array([False])]

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def reset(self):
        """
        Reset the line parameters to default values.
        """
        self.detected = False
        self.best_fit_coeffs = collections.deque(maxlen=2 * 10)
        self.recent_xfitted = [1, 1, 1]
        self.bestx = [1, 1, 1]
        self.current_fit_coeffs = [np.array([False])]
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

    def update(self, coeffs, is_detected=True, clear_buffer=False):
        """
        Update Line with new fitted coefficients.

        :param coeffs: new polynomial coefficients (in pixels)
        :param is_detected: if the Line was detected or inferred
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = is_detected

        if not self.detected:
            self.reset()

        if clear_buffer:
            self.recent_xfitted = []

        self.current_fit_coeffs = coeffs
        self.best_fit_coeffs.append(self.current_fit_coeffs)
        self.recent_xfitted.append(self.current_fit_coeffs)
        self.bestx = np.average(self.recent_xfitted, axis=0)

    def radius_of_curvature(self, coeffs, ploty):
        """
        Radius of curve = ( (1 + (2Ay + B)^2)^(3/2) ) / |2A|

        :param coeffs: polynomial coefficients
        :param ploty: y parameters for plotting
        :return: radius
        """
        A = coeffs[0]
        B = coeffs[1]
        y = np.max(ploty) * self.hyperparameters.lane().metres_per_pixel_y

        r_curve = ((1 + (2 * A * y + B) ** 2) ** (3 / 2)) / np.absolute(2 * A)
        return r_curve

    @property
    def average_fit(self):
        """
        :return: average of polynomial coefficients of the last N iterations
        """
        return np.mean(self.best_fit_coeffs, axis=0)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.current_fit_coeffs

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)
