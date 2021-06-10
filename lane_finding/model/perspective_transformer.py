import cv2
import numpy as np


class PerspectiveTransformer:

    def birdseye_transform(self, image):
        """
        Apply perspective transform to input frame to get the bird's eye view.
        :return: warped image, and both forward and backward transformation matrices
        """
        h, w = image.shape[:2]

        src = np.float32([[w, h - 10],  # br
                          [0, h - 10],  # bl
                          [546, 460],  # tl
                          [732, 460]])  # tr

        dst = np.float32([[w, h],  # br
                          [0, h],  # bl
                          [0, 0],  # tl
                          [w, 0]])  # tr

        # calculate the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)

        # Calculate the inverse
        Minv = cv2.getPerspectiveTransform(dst, src)

        # Create warped image - uses linear interpolation
        warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

        return warped, M, Minv
