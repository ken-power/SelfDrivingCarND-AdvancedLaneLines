class Hyperparameters:
    """
    Encapsulates the hyperparameters required for tuning the lane-finding code.
    """

    def __init__(self):
        # whether or not line state is maintained when building output images
        self.keep_state = True
        self.lane_parameters = LaneParams()
        self.thresholding_parameters = BinaryThresholdParams()
        self.image_frame_number = 0
        self.reset_lane_search = True

    def set_keep_state(self, keep_state):
        self.keep_state = keep_state

    def lane(self):
        return self.lane_parameters

    def thresholding(self):
        return self.thresholding_parameters


class LaneParams:
    """
    Hyperparameters related to lanes.
    """
    def __init__(self):
        # Assumptions for default values:
        # - the lane we are projecting is about 30 meters long and 3.7 meters wide
        # - our camera image has 720 relevant pixels in the y-dimension (remember, our image is perspective-transformed)
        # - our camera image has roughly 700 relevant pixels in the x-dimension
        # These assumptions lead to the following definitions:
        #
        # Define conversions in x and y from pixels space to meters
        self.metres_per_pixel_y = 30 / 720  # meters per pixel in y dimension
        self.metres_per_pixel_x = 3.7 / 700  # meters per pixel in x dimension

        # Number of windows to use for finding the lane line using sliding windows
        self.num_sliding_windows = 20

        # width of the windows +/- margin
        self.margin_first_frame = 100
        # minimum number of pixels found to recenter window
        self.minipix_first_frame = 50
        # The width of the margin around the previous polynomial to search
        self.margin_second_frame = 100
        self.minipix_second_frame = 50

        # Min and max values for projecting the discovered lane onto the road image
        self.min_lane_projection_width = 200
        self.max_lane_projection_width = 1000
        self.poly_fit_val = 100

    def set_meters_per_pixel_x(self, lane_projection_width_meters, pixels_in_x_direction):
        """
        Set meters per pixel in x dimension.

        :param lane_projection_width_meters: the width of the lane we are projecting on to (meters)
        :param pixels_in_x_direction: number of relevant image pixels in the x-dimension
        :return:
        """
        self.metres_per_pixel_x = lane_projection_width_meters / pixels_in_x_direction

    def set_meters_per_pixel_y(self, lane_projection_length_meters, pixels_in_y_direction):
        """
        Set meters per pixel in y dimension.

        :param lane_projection_length_meters: the length of the lane we are projecting on to (meters)
        :param pixels_in_y_direction: number of relevant image pixels in the y-dimension
        :return:
        """
        self.metres_per_pixel_y = lane_projection_length_meters / pixels_in_y_direction

    def set_num_windows(self, n):
        self.num_sliding_windows = n

    def set_margin_first_frame(self, m):
        self.margin_first_frame = m

    def set_minipix_first_frame(self, m):
        self.minipix_first_frame = m

    def set_margin_second_frame(self, m):
        self.margin_second_frame = m

    def set_minipix_second_frame(self, m):
        self.minipix_second_frame = m

    def set_lane_projection_width(self, min, max):
        self.min_lane_projection_width = min
        self.max_lane_projection_width = max

    def set_poly_fit_val(self, n):
        self.poly_fit_val = n


class BinaryThresholdParams:
    """
    Hyperparameters related to binary thresholding.
    """
    def __init__(self):
        # Color thresholding function
        self.thresh_function = 1
        self.sobel_thresh = (170, 255)
        self.sobelx_thresh = (20, 100)
        self.sobel_k_size = 3

    def thresholding_function(self):
        return self.thresh_function

    def set_thresholding_function(self, f):
        self.thresh_function = f

    def sobel_threshold(self):
        return self.sobel_thresh

    def set_sobel_threshold(self, min, max):
        self.sobel_thresh = (min, max)

    def sobelx_threshold(self):
        return self.sobelx_thresh

    def set_sobelx_threshold(self, min, max):
        self.sobelx_thresh = (min, max)

    def sobel_kernel_size(self):
        return self.sobel_k_size

    def set_sobel_kernel_size(self, k_size):
        self.sobel_k_size = k_size
