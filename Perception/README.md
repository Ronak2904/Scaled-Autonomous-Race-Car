# LANE DETECTION

This project follows a ROS-based Lane Detection pipeline. 
It takes in the images from the Intel Realsense T265 camera as a ros image.
![raw_image](https://user-images.githubusercontent.com/100716616/208428274-ed7302bf-c01f-4ced-a10c-da1b8cda36d5.png)

Since the acquired image is in a fisheye format, it is first sent through a fisheye undistortion. 

Then the resulting image is passed through a color threshold to get the lanes segregated from the neighbouring image. All of this is done in a  'bird's eye' view (as seen from the top).
![birds_eye](https://user-images.githubusercontent.com/100716616/208435928-ecfb182c-5cdf-45b6-a2f6-ee5381cf3e6e.png)


The histogram of pixel intensities is used to detect edges. The beginning of lane borders is determined by a sharp change in intensity. There are two such pixel intensity variations, one for the right lane boundary and one for the left lane boundary. The leftmost pixel intensity shift is supposed to be the start of the left lane border, while the other pixel intensity change is assumed to be the start of the right lane boundary.
![hist](https://user-images.githubusercontent.com/100716616/208437754-ba4075e5-be14-47f0-a05e-0bbba8708a40.png)

After distinctifying the right and left lanes, a sliding window approach is used to get the shape of the right and the left lanes, and a parabola is fitted to get the curvature as well as the offset of the car from the center of the lanes.
![lines](https://user-images.githubusercontent.com/100716616/208438858-a3b82ee0-7c17-4472-adf4-e96cfff12e0a.png)
