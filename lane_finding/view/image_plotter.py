import os

import cv2
import matplotlib.pyplot as plt


class ImagePlotter:

    def plot_image(self, img):
        plt.plot(img)

    def show_image(self, img):
        plt.imshow(img)
        plt.show()

    def plot_images(self, images, rows=5, cols=4):
        """
        'images' is a list that contains [image, filename] pairs
        """

        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
        axs = axs.ravel()

        for i, image in enumerate(images):
            # draw and display the corners
            axs[i].axis('off')
            axs[i].set_title(os.path.basename(image[1]))
            axs[i].imshow(image[0])

    def plot_2_images_side_by_side_BGR2RGB(self,
                                           img_1, img_1_title,
                                           img_2, img_2_title):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.tight_layout()

        ax1.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
        ax1.set_title(img_1_title, fontsize=25)

        ax2.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
        ax2.set_title(img_2_title, fontsize=25)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_4_images_side_by_side_BGR2RGB(self,
                                           img_1, img_1_title,
                                           img_2, img_2_title,
                                           img_3, img_3_title,
                                           img_4, img_4_title):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        f.tight_layout()

        ax1.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
        ax1.set_title(img_1_title, fontsize=25)

        ax2.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
        ax2.set_title(img_2_title, fontsize=25)

        ax3.imshow(cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB))
        ax3.set_title(img_3_title, fontsize=25)

        ax4.imshow(cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB))
        ax4.set_title(img_4_title, fontsize=25)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_4_images_side_by_side_cmap(self,
                                        img_1, img_1_title,
                                        img_2, img_2_title,
                                        img_3, img_3_title,
                                        img_4, img_4_title,
                                        cmap_img_1=None,
                                        cmap_img_2='gray',
                                        cmap_img_3=None,
                                        cmap_img_4=None
                                        ):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        f.tight_layout()

        ax1.set_title(img_1_title, fontsize=25)
        ax1.imshow(img_1, cmap=cmap_img_1)

        ax2.set_title(img_2_title, fontsize=25)
        ax2.imshow(img_2, cmap=cmap_img_2)

        ax3.set_title(img_3_title, fontsize=25)
        ax3.imshow(img_3, cmap=cmap_img_3)

        ax4.set_title(img_4_title, fontsize=25)
        ax4.imshow(img_4, cmap=cmap_img_4)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_5_images_side_by_side_cmap(self,
                                        img_1, img_1_title,
                                        img_2, img_2_title,
                                        img_3, img_3_title,
                                        img_4, img_4_title,
                                        img_5, img_5_title,
                                        cmap_img_1=None,
                                        cmap_img_2='gray',
                                        cmap_img_3='gray',
                                        cmap_img_4=None,
                                        cmap_img_5=None
                                        ):
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 10))
        f.tight_layout()
        font_size = 18

        ax1.set_title(img_1_title, fontsize=font_size)
        ax1.imshow(img_1, cmap=cmap_img_1)

        ax2.set_title(img_2_title, fontsize=font_size)
        ax2.imshow(img_2, cmap=cmap_img_2)

        ax3.set_title(img_3_title, fontsize=font_size)
        ax3.imshow(img_3, cmap=cmap_img_3)

        ax4.set_title(img_4_title, fontsize=font_size)
        ax4.imshow(img_4, cmap=cmap_img_4)

        ax5.set_title(img_5_title, fontsize=font_size)
        ax5.imshow(img_5, cmap=cmap_img_5)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_2_images_side_by_side_cmap(self,
                                        img_1, img_1_title,
                                        img_2, img_2_title,
                                        cmap_img_1=None,
                                        cmap_img_2=None):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.tight_layout()

        ax1.set_title(img_1_title, fontsize=25)
        ax1.imshow(img_1, cmap=cmap_img_1)

        ax2.set_title(img_2_title, fontsize=25)
        ax2.imshow(img_2, cmap=cmap_img_2)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_birseye_perspective_and_histogram(self,
                                               birdseye_transform, img_1_title,
                                               histogram, img_2_title,
                                               cmap_img_1=None,
                                               cmap_img_2=None):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.tight_layout()

        ax1.set_title(img_1_title, fontsize=25)
        ax1.imshow(birdseye_transform, cmap=cmap_img_1)

        ax2.set_title(img_2_title, fontsize=25)
        try:
            ax2.imshow(histogram)
        except:
            ax2.plot(histogram)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_3_images_side_by_side_cmap(self,
                                        img_1, img_1_title,
                                        img_2, img_2_title,
                                        img_3, img_3_title,
                                        cmap_img_1='gray',
                                        cmap_img_2='gray',
                                        cmap_img_3='gray'):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(img_1, cmap=cmap_img_1)
        ax1.set_title(img_1_title, fontsize=50)

        ax2.imshow(img_2, cmap=cmap_img_2)
        ax2.set_title(img_2_title, fontsize=50)

        ax3.imshow(img_3, cmap=cmap_img_3)
        ax3.set_title(img_3_title, fontsize=50)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_6_images_in_one_line(self, img, eq_white_mask, hsv_yellow_mask, sobel_mask, binary, closing):
        f, ax = plt.subplots(1, 6, figsize=(24, 9))

        f.set_facecolor('white')
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title('input_frame')
        ax[0].set_axis_off()
        # ax[0, 0].set_axis_bgcolor('red')

        ax[1].imshow(eq_white_mask, cmap='gray')
        ax[1].set_title('white mask')
        ax[1].set_axis_off()

        ax[2].imshow(hsv_yellow_mask, cmap='gray')
        ax[2].set_title('yellow mask')
        ax[2].set_axis_off()

        ax[3].imshow(sobel_mask, cmap='gray')
        ax[3].set_title('sobel mask')
        ax[3].set_axis_off()

        ax[4].imshow(binary, cmap='gray')
        ax[4].set_title('before closure')
        ax[4].set_axis_off()

        ax[5].imshow(closing, cmap='gray')
        ax[5].set_title('after closure')
        ax[5].set_axis_off()

        plt.show()

    def plot_4_rows_of_2_images_cmap(self,
                                     img_1, img_1_title,
                                     img_2, img_2_title,
                                     img_3, img_3_title,
                                     img_4, img_4_title,
                                     img_5, img_5_title,
                                     img_6, img_6_title,
                                     img_7, img_7_title,
                                     img_8, img_8_title,
                                     cmap_img_1='gray',
                                     cmap_img_2='gray',
                                     cmap_img_3='gray',
                                     cmap_img_4='gray',
                                     cmap_img_5='gray',
                                     cmap_img_6='gray',
                                     cmap_img_7='gray',
                                     cmap_img_8='gray'):
        font = {'size': 30}

        plt.rc('font', **font)

        plt_image_height = 35
        plt_image_width = 55

        # Plot the result
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2,
                                                                           figsize=(plt_image_height, plt_image_width))
        f.tight_layout(pad=5)

        ax1.imshow(img_1, cmap=cmap_img_1)
        ax1.set_title(img_1_title)

        ax2.imshow(img_2, cmap=cmap_img_2)
        ax2.set_title(img_2_title)

        ax3.imshow(img_3, cmap=cmap_img_3)
        ax3.set_title(img_3_title)

        ax4.imshow(img_4, cmap=cmap_img_4)
        ax4.set_title(img_4_title)

        ax5.imshow(img_5, cmap=cmap_img_5)
        ax5.set_title(img_5_title)

        ax6.imshow(img_6, cmap=cmap_img_6)
        ax6.set_title(img_6_title)

        ax7.imshow(img_7, cmap=cmap_img_7)
        ax7.set_title(img_7_title)

        ax8.imshow(img_8, cmap=cmap_img_8)
        ax8.set_title(img_8_title)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def plot_2_rows_of_2_images_cmap(self,
                                     img_1, img_1_title,
                                     img_2, img_2_title,
                                     img_3, img_3_title,
                                     img_4, img_4_title,
                                     cmap_img_1='gray',
                                     cmap_img_2='gray',
                                     cmap_img_3='gray',
                                     cmap_img_4='gray'):
        font = {
            'size': 30}

        plt.rc('font', **font)

        plt_image_height = 40
        plt_image_width = 55

        # Plot the result
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                   figsize=(plt_image_height, plt_image_width))
        f.tight_layout(pad=2)

        ax1.imshow(img_1, cmap=cmap_img_1)
        ax1.set_title(img_1_title)

        ax2.imshow(img_2, cmap=cmap_img_2)
        ax2.set_title(img_2_title)

        ax3.imshow(img_3, cmap=cmap_img_3)
        ax3.set_title(img_3_title)

        ax4.imshow(img_4, cmap=cmap_img_4)
        ax4.set_title(img_4_title)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
