import numpy as np
import cv2
import rospy
import matplotlib.pyplot as plt
import shutil
from utils import *
from collections import deque
from sensor_msgs.msg import CompressedImage,Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import math

global total_img_count
total_img_count = 0

def create_queue(length = 1):
    return deque(maxlen=length)

mtx = np.array([[ 286.221588134766,             0,  419.467010498047],
                [            0, 287.480102539062, 386.97509765625],
                [            0,             0,              1]])

dist = np.array([-0.0043481751345098, 0.037125650793314, -0.0355393998324871, 0.00577297387644649])

rospy.init_node('warpedimage', anonymous=True)
#rospy.loginfo("Hello Fisheye Image")

bridge = CvBridge()


def undistort(img):
    global mtx,dist
    result = cv2.fisheye.undistortImage(img, mtx, dist, None, mtx)
    return result


  
def compute_hls_white_yellow_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
 

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_yellow_bin = np.zeros_like(rgb_img) 
 
    x2 = ((rgb_img >= 230) & (rgb_img <= 255))
    img_hls_white_yellow_bin[x2] = 1

 
    return img_hls_white_yellow_bin


def compute_perspective_transform_matrices(src, dst):
    """
    Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
    and M_inv is the matrix used to revert the transformed image back to the original one
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return (M, M_inv)

   


def draw_lane_lines(warped_img, left_fitx, right_fitx,win_l, win_r):
    """
    Returns an image where the computed lane lines have been drawn on top of the original warped binary image
    """
    # Create an output image with 3 colors (RGB) from the binary warped image to draw on and  visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img))*255

    # Now draw the lines
    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
    pts_left = np.dstack((left_fitx, ploty)).astype(np.int32)
    pts_right = np.dstack((right_fitx, ploty)).astype(np.int32)

    cv2.polylines(out_img, pts_left, False,  (255, 140,0), 5)
    cv2.polylines(out_img, pts_right, False, (255, 140,0), 5)

    for low_pt, high_pt in win_l:
        cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

    for low_pt, high_pt in win_r:
        cv2.rectangle(out_img, low_pt, high_pt, (0, 255, 0), 3)

    return out_img

  
def compute_lane_curvature(polynomial_coeff_l, polynomial_coeff_r, ploty, lane_center_px_psp, xm_per_px):
    """
    Returns the triple (left_curvature, right_curvature, lane_center_offset), which are all in meters
    """
    y_eval = np.max(ploty)-450
    # Define conversions in x and y from pixels space to meters

    #leftx = left_line.line_fit_x
    #rightx = right_line.line_fit_x

    # Fit new polynomials: find x for y in real-world space
    # left_fit_cr = np.polyfit(ploty * self.ym_per_px, leftx * self.xm_per_px, 2)
    #right_fit_cr = np.polyfit(ploty * self.ym_per_px, rightx * self.xm_per_px, 2)

    # Now calculate the radii of the curvature
    #left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    #right_curverad = ((1 + (2 *right_fit_cr[0] * y_eval * self.ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    # Use our computed polynomial to determine the car's center position in image space, then
    left_fit = polynomial_coeff_l
    right_fit = polynomial_coeff_r

    center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) +
                                (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - lane_center_px_psp
    center_offset_real_world_m = center_offset_img_space * xm_per_px

    # Now our radius of curvature is in meters
    #return left_curverad, right_curverad, center_offset_real_world_m
    return center_offset_real_world_m



def compute_lane_lines(warped_img, sliding_windows_per_line, sliding_window_half_width, sliding_window_recenter_thres):
    """
    Returns the tuple (left_lane_line, right_lane_line) which represents respectively the LaneLine instances for
    the computed left and right lanes, for the supplied binary warped image
    """

    # Take a histogram of the bottom half\forth of the image, summing pixel values column wise
    histogram = np.sum(warped_img[(warped_img.shape[0]- 80):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    lane_base_1 = np.argmax(histogram)
    #histogram_1 = np.delete(histogram, range(lane_base_1-30,lane_base_1+30,1))
    histogram[lane_base_1-30:lane_base_1+30] = 0
    lane_base_2 = np.argmax(histogram)
    #leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint # don't forget to offset by midpoint!
    leftx_base = min(lane_base_1, lane_base_2)
    rightx_base = max(lane_base_1, lane_base_2)

    #print("right lane pct={0}".format(rightx_base))

    # Set height of windows
    window_height = np.int(warped_img.shape[0]//sliding_windows_per_line)
    # Identify the x and y positions of all nonzero pixels in the image
    # NOTE: nonzero returns a tuple of arrays in y and x directions
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    total_non_zeros = len(nonzeroy)
    non_zero_found_pct = 0.0

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base


    # Set the width of the windows +/- margin
    margin = sliding_window_half_width
    # Set minimum number of pixels found to recenter window
    minpix = sliding_window_recenter_thres
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Our lane line objects we store the result of this computation
    """

    if self.previous_left_lane_line is not  None and self.previous_right_lane_line is not None:
        x3 = (self.previous_left_lane_line.polynomial_coeff[0] * (nonzeroy**2)
                                       + self.previous_left_lane_line.polynomial_coeff[1] * nonzeroy
                                       + self.previous_left_lane_line.polynomial_coeff[2])
        # We have already computed the lane lines polynomials from a previous image
        left_lane_inds = ((nonzerox > (x3 - margin))
                          & (nonzerox < (x3+ margin)))
        x4 = (self.previous_right_lane_line.polynomial_coeff[0] * (nonzeroy**2)
                                        + self.previous_right_lane_line.polynomial_coeff[1] * nonzeroy
                                        + self.previous_right_lane_line.polynomial_coeff[2])

        right_lane_inds = ((nonzerox > (x4 - margin))
                           & (nonzerox < (x4 + margin)))

        non_zero_found_left = np.sum(left_lane_inds)
        non_zero_found_right = np.sum(right_lane_inds)
        non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
        #non_zero_found_pct = 0.8
        print("[Previous lane] Found pct={0}".format(non_zero_found_pct))
        #print(left_lane_inds)
        """


    #if non_zero_found_pct < 0.8:
    #print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
    left_lane_inds = []
    right_lane_inds = []
    win_l = []
    win_r = []

    # Step through the windows one by one
    for window in range(sliding_windows_per_line):
        # Identify window boundaries in x and y (and right and left)
        # We are moving our windows from the bottom to the top of the screen (highest to lowest y value)
        win_y_low = warped_img.shape[0] - (window + 1)* window_height
        win_y_high = warped_img.shape[0] - window * window_height

        # Defining our window's coverage in the horizontal (i.e. x) direction
        # Notice that the window's width is twice the margin
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        win_l.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
        win_r.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

        # Super crytic and hard to understand...
        # Basically nonzerox and nonzeroy have the same size and any nonzero pixel is identified by
        # (nonzeroy[i],nonzerox[i]), therefore we just return the i indices within the window that are nonzero
        # and can then index into nonzeroy and nonzerox to find the ACTUAL pixel coordinates that are not zero
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices since we now have a list of multiple arrays (e.g. ([1,3,6],[8,5,2]))
        # We want to create a single array with elements from all those lists (e.g. [1,3,6,8,5,2])
        # These are the indices that are non zero in our sliding windows
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    non_zero_found_left = np.sum(left_lane_inds)
    non_zero_found_right = np.sum(right_lane_inds)
    non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros

    # print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))



    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #print("[LEFT] Number of hot pixels={0}".format(len(leftx)))
    #print("[RIGHT] Number of hot pixels={0}".format(len(rightx)))
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #print("Poly left {0}".format(left_fit))
    #print("Poly right {0}".format(right_fit))
    polynomial_coeff_l = left_fit
    polynomial_coeff_r = right_fit
    """

    if  self.previous_left_lane_lines.append(left_line):
        left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
        left_line.polynomial_coeff = left_fit
        self.previous_left_lane_lines.append(left_line, force=True)
       # print("**** REVISED Poly left {0}".format(left_fit))
    #else:
        #left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
        #left_line.polynomial_coeff = left_fit


    if  self.previous_right_lane_lines.append(right_line):
        right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
        right_line.polynomial_coeff = right_fit
        self.previous_right_lane_lines.append(right_line, force=True)
        #print("**** REVISED Poly right {0}".format(right_fit))
    #else:
        #right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
        #right_line.polynomial_coeff = right_fit
        """



    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0] )
    #left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]


    #return (histogram,left_line, right_line)
    #return (polynomial_coeff_l, polynomial_coeff_r, left_fitx , right_fitx, win_l, win_r, ploty)
    return (polynomial_coeff_l, polynomial_coeff_r,ploty)

def process_image(img):
    """
    Attempts to find lane lines on the given image and returns an image with lane area colored in green
    as well as small intermediate images overlaid on top to understand how the algorithm is performing
    """
    # First step - undistort the image using the instance's object and image points
    #self.img = img

    global total_img_count
    #img = undistort(img)

    # Produce binary thresholded image from color and gradients
    
    hls_w_y_thres = compute_hls_white_yellow_binary(img)
    M_psp, Minv_psp =  compute_perspective_transform_matrices(src_pts, dst_pts)
    # Create the undistorted and binary perspective transforms
    img_size = (840, 800)
    #undist_img_psp = cv2.warpPerspective(undist_img, self.M_psp, img_size, flags=cv2.INTER_LINEAR)
    #print(M_psp)
    thres_img_psp = cv2.warpPerspective(hls_w_y_thres , M_psp, img_size, flags=cv2.INTER_LINEAR)
    #img_psp = warpPerspective(img,self.M_psp, img_size, flags=INTER_LINEAR)
    #hist, ll, rl = self.compute_lane_lines(thres_img_psp)
    #t.tic()
    ll, rl,ploty = compute_lane_lines(thres_img_psp, 3, 40, 40)
    #t.toc()
    #print(wl,wr)
    #print(ll.line_fit_x,rl.line_fit_x)
    #lcr, rcr, lco = self.compute_lane_curvature(ll, rl)
    lco = compute_lane_curvature(ll, rl, ploty, 445, 0.00135)

    # These lines are commented out for fast processing, uncomment to view the klane plotted images
    #drawn_lines = draw_lane_lines(thres_img_psp, lx, rx, wl, wr)
    #plt.imshow(drawn_lines)

    #drawn_lines_regions = self.draw_lane_lines_regions(thres_img_psp, ll, rl)
    #plt.imshow(drawn_lines_regions)

    #drawn_lane_area = self.draw_lane_area(thres_img_psp, img, ll, rl)
    #plt.imshow(drawn_lane_area)


    #drawn_hotspots = self.draw_lines_hotspots(thres_img_psp, ll, rl)
    #plt.imshow(drawn_hotspots)
    #combined_lane_img = self.combine_images(drawn_lane_area, drawn_lines, drawn_lines_regions, drawn_hotspots, img)
    #plt.imshow(combined_lane_img)
    #plt.show()
    #final_img = drawn_lines#self.draw_lane_curvature_text(drawn_lines, lcr, rcr, lco)


    total_img_count += 1
    previous_left_lane_line = ll
    previous_right_lane_line = rl
    #sxy_combined_dir_1 = np.uint8(255 * sxy_combined_dir)
    #hls_w_y_thres_1 = np.uint8(255 * hls_w_y_thres)


    return lco,thres_img_psp*255
    #return lco
    #return lco,img_psp
    #return thres_img_psp*255
    #return hist,thres_img_psp*255
    #return final_img

# Horizontal = 848, Vertical = 800
img_size = (848, 800)
offset = 0
pts = src = np.array([
        [0, 480],  #bottom-left corner
        [0, 445],  #top-left corner
        [840, 445], # top-right corner
        [840, 480], # bottom-right corner
    ], np.int32)

src_pts = pts.astype(np.float32)

# [ col , row] 
dst_pts = np.float32([
        [offset, 800],             # bottom-left corner
        [offset, 0],                       # top-left corner
        [840-offset, 0],           # top-right corner
        [840-offset, 800]  # bottom-right corner
    ])



 
def image_callback_1(img):

        cv_image = bridge.imgmsg_to_cv2(img)
        
        center_offset,img_psp = process_image(cv_image)
        #print(ll)
        #=histo,img_= ld.process_image(cv_image)
        #plt.imshow(img_psp, cmap = 'gray')
        #plt.show()
        #plt.plot(histo)
        #cv2.imshow('proc_img',img_psp)
        #cv2.waitKey(1)
        #imtransf = bridge.cv2_to_imgmsg(img_psp)
        #print(np.max(proc_img))
        #pub_image.publish(imtransf)
        #pub_psp.publish(impsp)
        #pub_leftcurvature.publish(left_curvature)
        #pub_rightcurvature.publish(right_curvature)
        pub_offset.publish(center_offset)



sub_image = rospy.Subscriber("/camera/fisheye2/image_raw", Image, image_callback_1)
#pub_image = rospy.Publisher("lanedetectedimage", Image, queue_size = 1)
pub_offset = rospy.Publisher("offset", Float32, queue_size=10)
#pub_leftcurvature = rospy.Publisher("lcurvature", Float32, queue_size=10)
#pub_rightcurvature = rospy.Publisher("rcurvature", Float32, queue_size=10)
rate = rospy.Rate(200)


while not rospy.is_shutdown():
    cv2.destroyAllWindows()
    rospy.spin()
