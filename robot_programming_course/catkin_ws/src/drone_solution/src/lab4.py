#!/usr/bin/env python
# coding: utf-8

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from hector_uav_msgs.srv import EnableMotors

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from nav_msgs.msg import Odometry
import tf2_ros
import tf.transformations as ttt

#PLANING_HORIZON = 50
#TIME_LIFTOFF = 3

RING_AVOIDANCE_TIME = 2
DEFAULT_ALTITUDE = 3


Kp_z = 3
Kd_z = 6

Kp_y = 0.01
Kd_y = 0.00005

Kp_w = -0.01
Kd_w = -0.00005


class SimpleMover():

    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.cv_bridge = CvBridge()

        rospy.on_shutdown(self.shutdown)

        rospy.Subscriber("cam_1/camera/image", Image, self.camera_callback)

        rospy.Subscriber("cam_2/camera/image", Image, self.camera_rings_callback)

        rospy.Subscriber('/ground_truth/state', Odometry, self.odom_callback)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.drone_state = [0] * 6  # position vector
        self.y_error = 0
        self.omega_error = 0
        self.z_des = DEFAULT_ALTITUDE
        self.image_1 = []
        self.image_2 = []
        self.state = "free_flight"
        self.red_ring_detected = False
        self.blue_ring_detected = False
        self.time_start_up = 0
        self.avoidance_time = 0
        self.e_x_blue, self.e_y_blue = 0, 0
        self.rate = rospy.Rate(30)

    def odom_callback(self, msg):
        """ Pose of a robot extraction"""
        transform = self.tfBuffer.lookup_transform('world', 'base_stabilized', rospy.Time()).transform
        x, y, z = transform.translation.x, transform.translation.y, transform.translation.z
        quat = transform.rotation
        r, p, y = ttt.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        self.drone_state = [x, y, z, r, p, y]

    def camera_callback(self, msg):
        """Computer vision stuff"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Эффективное преобразование и пороговое применение
        grey_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey_image, 8, 255, cv2.THRESH_BINARY_INV)

        # Предварительно рассчитываемые значения
        height_half = msg.height // 2
        width_half = msg.width // 2

        # Векторизованное нахождение точек
        top_indices = np.flatnonzero(mask[10] >= 10)
        mid_indices = np.flatnonzero(mask[height_half] >= 10)

        if top_indices.size and mid_indices.size:
            top_line_point = int(np.mean(top_indices))
            mid_line_point = int(np.mean(mid_indices))
            self.omega_error = top_line_point - mid_line_point

            # Визуализация точек и линии направления
            cv2.circle(cv_image, (top_line_point, 10), 5, (0, 0, 255), -1)
            cv2.circle(cv_image, (mid_line_point, height_half), 5, (0, 0, 255), -1)
            cv2.line(cv_image, (mid_line_point, height_half), (top_line_point, 10), (0, 0, 255), 3)

        # y-offset control
        y_indices = np.flatnonzero(mask[height_half] >= 10)
        if y_indices.size:
            cy = int(np.mean(y_indices))
            self.y_error = width_half - cy

            # Визуализация смещения по y
            cv2.circle(cv_image, (cy, height_half), 7, (0, 255, 0), -1)
            cv2.line(cv_image, (width_half, height_half), (cy, height_half), (0, 255, 0), 3)

        # Если вы хотите отображать изображение (в режиме отладки)
        # cv2.imshow("Camera View", cv_image)
        # cv2.waitKey(3)

        self.image_1 = cv_image

    def camera_rings_callback(self, msg):
        # """ Computer vision stuff for Rings"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # red
        lower = np.uint8([0, 0, 90])
        upper = np.uint8([30, 30, 120])
        cv_image, red_pose, red_radius = self.ring_detector(cv_image, lower, upper, (0, 0, 255))

        # blue
        lower = np.uint8([40, 20, 20])
        upper = np.uint8([80, 50, 50])
        cv_image, blue_pose, blue_radius = self.ring_detector(cv_image, lower, upper, (255, 0, 0))

        # print(red_radius, blue_radius)

        if 50 < red_radius < 70 or 50 < blue_radius < 80:
            if red_radius > blue_radius:
                self.blue_ring_detected = False
                self.red_ring_detected = True
            else:
                self.red_ring_detected = False
                self.blue_ring_detected = True

                # offset in ring xy-plane to fly through center of a ring
                # error = <center of image> - <center of ring>
                self.e_x_blue = 160 - blue_pose[0]
                self.e_y_blue = 120 - blue_pose[1]
        else:
            self.blue_ring_detected = False
            self.red_ring_detected = False

        # save results
        self.image_2 = cv_image

    def ring_detector(self, image, lower, upper, color):
     color_mask = cv2.inRange(image, lower, upper)
     color_contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     if not color_contours:
        return image, (0, 0), 0

     max_contour_idx = max(range(len(color_contours)), key=lambda i: len(color_contours[i]))
     c = color_contours[max_contour_idx]
     M = cv2.moments(c)

     if M['m00'] > 1e-5:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        (x, y), radius = cv2.minEnclosingCircle(c)

        if radius > 10:
            image = cv2.circle(image, (cx, cy), radius=5, color=color, thickness=-1)
            cv2.drawContours(image, [c], -1, color, 1)
            image = cv2.circle(image, (int(x), int(y)), radius=int(radius), color=color, thickness=4)
            return image, (x, y), radius

     return image, (0, 0), 0

    def fsm_update(self):
        if self.red_ring_detected:
            self.state = "drone_up"
            self.time_start_up = 0
        elif RING_AVOIDANCE_TIME < self.avoidance_time < RING_AVOIDANCE_TIME + 4:
            self.state = "drone_down"
        elif self.blue_ring_detected:
            self.state = "drone_blue_ring"
        else:
            self.state = "free_flight"

    def show_image(self, img, title='Camera 1'):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def enable_motors(self):
        try:
            rospy.wait_for_service('enable_motors', 2)
            call_service = rospy.ServiceProxy('enable_motors', EnableMotors)
            response = call_service(True)
        except Exception as e:
            print("Error while try to enable motors: ", e)

    def spin(self):
        self.enable_motors()

        # Initialisations
        altitude_prev = 0
        y_error_prev = 0
        omega_error_prev = 0

        time_start = rospy.get_time()
        time_prev = time_start
        while not rospy.is_shutdown():
            try:
                # Time stuff
                t = rospy.get_time() - time_start
                dt = t - time_prev
                time_prev = t
                if dt == 0:
                    dt = 1 / 30.0

                # Write here altitude controller
                u_z = Kp_z * (self.z_des - self.drone_state[2]) - Kd_z * (self.drone_state[2] - altitude_prev) / dt
                altitude_prev = self.drone_state[2]

                # Steering control
                u_omega_z = Kp_w * self.omega_error - Kd_w * (self.omega_error - omega_error_prev) / dt
                omega_error_prev = self.omega_error

                # Offset control
                u_y = Kp_y * self.y_error - Kd_y * (self.y_error - y_error_prev) / dt
                y_error_prev = self.y_error

                self.fsm_update()
                if self.state == "drone_up":
                    self.z_des = 5
                if self.time_start_up == 0:
                    self.time_start_up = rospy.get_time()
                elif self.state == "drone_down":
                    self.z_des = DEFAULT_ALTITUDE
                elif self.state == "drone_blue_ring":
                    self.z_des += 0.001 * self.e_y_blue
                    if self.z_des < 2:
                        self.z_des = 2
                    pass
                elif self.state == "free_flight":
                    pass
                else:
                    rospy.logerr("Error: state name error!")

                self.avoidance_time = rospy.get_time() - self.time_start_up

                print(self.state, self.z_des)

                # Display augmented images from both cameras
                if len(self.image_1) > 0 and len(self.image_2) > 0:
                    self.show_image(self.image_1, title='Line')
                    self.show_image(self.image_2, title='Rings')

                twist_msg = Twist()
                twist_msg.linear.x = 1.0
                twist_msg.linear.y = u_y
                twist_msg.linear.z = u_z
                twist_msg.angular.z = u_omega_z
                self.cmd_vel_pub.publish(twist_msg)

            except KeyboardInterrupt:
                break

            self.rate.sleep()

    def shutdown(self):
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


if __name__ == "__main__":
    simple_mover = SimpleMover()
    simple_mover.spin()