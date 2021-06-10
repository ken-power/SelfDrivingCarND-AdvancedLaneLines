import cv2
import numpy as np


class ImageBuilder:

    def draw_lane_on_road(self, image, Minv, left_lane_line, right_lane_line, keep_state=True):
        """
        Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.
        :param image: original undistorted color frame
        :param Minv: (inverse) perspective transform matrix used to re-project on original frame
        :param left_lane_line: left lane-line previously detected
        :param right_lane_line: right lane-line previously detected
        :param keep_state: if True, line state is maintained
        :return: image with lane area overlaid on road image
        """
        height, width, _ = image.shape

        left_fit = left_lane_line.average_fit if keep_state else left_lane_line.current_fit_coeffs
        right_fit = right_lane_line.average_fit if keep_state else right_lane_line.current_fit_coeffs

        # Generate x and y values for plotting
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # draw road as green polygon on original frame
        road_warp = np.zeros_like(image, dtype=np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
        road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

        lane_area_on_road_img = cv2.addWeighted(image, 1., road_dewarped, 0.3, 0)

        # now separately draw solid lines to highlight them
        line_warp = np.zeros_like(image)
        line_warp = left_lane_line.draw(line_warp, color=(255, 0, 0), average=keep_state)
        line_warp = right_lane_line.draw(line_warp, color=(0, 0, 255), average=keep_state)
        line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

        lines_mask = lane_area_on_road_img.copy()
        idx = np.any([line_dewarped != 0][0], axis=2)
        lines_mask[idx] = line_dewarped[idx]

        lane_area_on_road_img = cv2.addWeighted(src1=lines_mask, alpha=0.8,
                                                src2=lane_area_on_road_img, beta=0.5, gamma=0.)

        return lane_area_on_road_img

    def add_overlays_to_main_image(self,
                                   main_image,
                                   binary_image,
                                   birdseye_image,
                                   birdseye_lanes_image,
                                   undistorted_image,
                                   radius_of_curvature_metres,
                                   offset_in_meters,
                                   offset_direction,
                                   frame_num):
        """
        Create the final combined output image, using the intermediate pipeline images
        :return: combined output image with insets and text
        """

        main_image = self.add_inset_images(main_image,
                                           binary_image,
                                           birdseye_image,
                                           birdseye_lanes_image,
                                           undistorted_image)

        main_image = self.add_image_captions(main_image,
                                             "undistorted",
                                             "binary",
                                             "birdseye",
                                             "birdseye showing lanes")

        main_image = self.add_text_to_image(main_image,
                                            radius_of_curvature_metres,
                                            offset_in_meters,
                                            offset_direction)

        main_image = self.add_frame_data_to_image(main_image,
                                                  frame_num)

        return main_image

    def add_inset_images(self,
                         main_img,
                         binary_img,
                         birdseye_img,
                         birdseye_lanes_img,
                         undistorted_img):
        h, w = main_img.shape[:2]
        inset_ratio = 0.2
        inset_h, inset_w = int(inset_ratio * h), int(inset_ratio * w)
        x_offset, y_offset = 50, 30

        # add a rectangle to the upper area of the image
        mask = main_img.copy()
        mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, inset_h + 4 * y_offset), color=(0, 0, 0), thickness=cv2.FILLED)
        main_img = cv2.addWeighted(src1=mask, alpha=0.2, src2=main_img, beta=0.8, gamma=0)

        # add undistorted image
        undistorted_inset = cv2.resize(undistorted_img, dsize=(inset_w, inset_h))
        main_img[y_offset:inset_h + y_offset, x_offset:x_offset + inset_w, :] = undistorted_inset

        # add inset of binary image
        binary_inset = cv2.resize(binary_img, dsize=(inset_w, inset_h))
        binary_inset = np.dstack([binary_inset, binary_inset, binary_inset]) * 255
        main_img[y_offset:inset_h + y_offset, 2 * x_offset + inset_w:2 * (x_offset + inset_w), :] = binary_inset

        # add inset of bird's eye perspective
        birdseye_inset = cv2.resize(birdseye_img, dsize=(inset_w, inset_h))
        birdseye_inset = np.dstack([birdseye_inset, birdseye_inset, birdseye_inset]) * 255
        main_img[y_offset:inset_h + y_offset, 3 * x_offset + 2 * inset_w:3 * (x_offset + inset_w), :] = birdseye_inset

        # add inset of bird's eye view with highlighted lane lines
        birdseye_lanes_inset = cv2.resize(birdseye_lanes_img, dsize=(inset_w, inset_h))
        main_img[y_offset:inset_h + y_offset, 4 * x_offset + 3 * inset_w:4 * (x_offset + inset_w),
        :] = birdseye_lanes_inset

        return main_img

    def add_text_to_image(self,
                          main_img,
                          radius_of_curvature_metres,
                          offset_in_meters,
                          offset_direction):
        """
        Add text to the main image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_color = (255, 0, 255)
        font_thickness = 1
        line_type = cv2.LINE_AA

        # coordinates of the bottom-left corner of the text string in the image
        x_coord = 420
        y_coord = 180
        text_height_with_spacing = 35

        cv2.putText(main_img,
                    'Radius of curvature: {:.02f}m'.format(radius_of_curvature_metres),
                    (x_coord, y_coord + text_height_with_spacing),
                    font, font_scale, font_color, font_thickness, line_type)

        cv2.putText(main_img, 'Offset from center: {:.02f}m {}'.format(offset_in_meters, offset_direction),
                    (x_coord, y_coord + (2 * text_height_with_spacing)),
                    font, font_scale, font_color, font_thickness, line_type)

        return main_img

    def add_frame_data_to_image(self,
                                main_img,
                                frame_num):
        """
        Add text to the main image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 0, 255)
        font_thickness = 1
        line_type = cv2.LINE_AA

        # coordinates of the bottom-left corner of the text string in the image
        x_coord = 50
        y_coord = 220
        text_height_with_spacing = 15

        cv2.putText(main_img,
                    'Frame: {}'.format(frame_num),
                    (x_coord, y_coord + text_height_with_spacing),
                    font, font_scale, font_color, font_thickness, line_type)

        return main_img

    def add_image_captions(self,
                           main_img,
                           caption_1="caption 1",
                           caption_2="caption 2",
                           caption_3="caption 3",
                           caption_4="caption 4"):
        """
        Add text to the main image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_color = (255, 0, 255)
        font_thickness = 1
        line_type = cv2.LINE_AA

        # coordinates of the bottom-left corner of the text string in the image
        x_coord = 100
        y_coord = 25
        text_height_with_spacing = 35
        x_spacing = 200

        cv2.putText(main_img,
                    caption_1,
                    (x_coord, y_coord),
                    font, font_scale, font_color, font_thickness, line_type)

        cv2.putText(main_img,
                    caption_2,
                    (x_coord + int((1.7 * x_spacing)), y_coord),
                    font, font_scale, font_color, font_thickness, line_type)

        cv2.putText(main_img,
                    caption_3,
                    (x_coord + int((3.2 * x_spacing)), y_coord),
                    font, font_scale, font_color, font_thickness, line_type)

        cv2.putText(main_img,
                    caption_4,
                    (x_coord + int((4.3 * x_spacing)), y_coord),
                    font, font_scale, font_color, font_thickness, line_type)

        return main_img
