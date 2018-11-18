#!/usr/bin/env python
"""
Submission: Hungry Bird Team 2334

This python file runs a ROS-node of name drone_control which holds the position of e-Drone on the given target dummy.

This node publishes and subscribes the following topics:
        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command          /whycon/poses
        /alt_error              /pid_tuning_altitude
        /pitch_error            /pid_tuning_pitch
        /roll_error             /pid_tuning_roll
        /yaw_error              /pid_tuning_yaw
                                /drone_yaw

Code Description:
This file contains the Edrone class which is used to send commands to simulated drone through ROS interface by
publishing to the drone_command topic.It also publishes all the error topics to report the error in all the axes.
It subscribes the whycon/poses and drone_yaw topics to obtain the drone coordinates and orientation respectively.
It also subscribes to pid_tuning topics to change the pid values externally.

How it works:
Edrone receives real time coordinates through the respective callbacks in the background.Then it also runs a continuous
loop in the reach_target function where it calculates the error and the pid response values.The pid algorithm is
modified such that the integral coefficient only appears in a set range of errors in order to avoid reset integral
windup. Also the pid output response is clipped within a set of predetermined maximum and minimum values. We used numpy
internally to avoid code reduplication.
"""

from __future__ import print_function, division

import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray
from pid_tune.msg import PidTune
from plutodrone.msg import *
from std_msgs.msg import Float64

# set global printing options for numpy
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class Edrone:
    """
    Class corresponding to the E-Drone
    It is used to control the simulated or real E-Drone using an inbuilt pid controller.
    Commands are sent to the drone which continues following one command till the next is sent.

    Usage:

    >>> my_drone = Edrone()     # First disarmed for safety reasons
    Disarmed
    Armed
    Initialized Drone

    >>> my_drone.publish_command(throttle = 1550) #base value is 1500
    <Publisher Output>

    >>> my_drone.publish_command(roll=1550)  #base value is 1500
    <Publisher Output>

    >>> my_drone.publish_command() # Return the drone to equilibrium
    <Publisher Output>

    >>> my_drone.reach_target(target_pose) # Hover Near A target
    <Publisher outputs>

    >>> my_drone.disarm()  # Disarm Drone
    Disarmed

    >>> my_drone.arm() # Ready For Action
    Armed

    >>> my_drone.land()
    Disarmed


    Regarding Coordinate Systems:
    Internally the coordinates or poses of the drone are represented as numpy arrays in the order given by the
    sense_axes and control_axes variable. This ordering does not change.
    The order of the input coordinates is dictated by the cartesian_axes variable which may be changed if there is a mismatch
    in the physical orientation of coordinate systems. Currently 'pitch' corresponds to x-axis and so on.
    """

    # Ordering of values in numpy arrays used internally to represent pose.This order should not be changed.
    sense_axes = ('pitch', 'roll', 'altitude', 'yaw')
    control_axes = ('pitch', 'roll', 'throttle', 'yaw')

    # You can change the order here if there is a mismatch in coordinate
    # systems. Use x* etc to give negative axis
    cartesian_axes = ('x*', 'y*', 'z', 'yaw')

    def __init__(self):
        """
        Initialize the drone
        Automatically arm and bring the drone to a usable state
        Also initialize all the subscriptions and publications of the node
        Some initial values for the PID controller are defined here
        """
        rospy.init_node('drone_control')

        # CONTROLLER CONSTANTS
        # To be read as ['pitch', 'roll', 'altitude', 'yaw']
        self.sample_time = 0.10
        self.Kp = np.array([30, 30, 40, 0])
        self.Ki = np.array([70, 70, 50, 0])
        self.Kd = np.array([0, 0, 350, 0])
        self.max_Ki_margin = np.array([.05, .05, .04, 0.0])
        self.min_Ki_margin = np.array([0.02, 0.02, .02, 0.0])
        self.max_values = np.array([1515, 1515, 1800, 1800])
        self.min_values = np.array([1485, 1485, 1200, 1200])
        self.base_values = np.array([1500, 1500, 1500, 1500])

        # STATE VARIABLES
        self.setpoint = np.zeros(4)
        self.drone_position = np.zeros(4)
        self.current_time = time.time()
        self.previous_time = time.time()
        self.previous_error = np.zeros(4)
        self.cumulative_error = np.zeros(4)
        self.error = np.zeros(4)

        # I/O INITIALIZATION
        self.cmd_msg = PlutoMsg()
        self.error_msg = Float64()

        # PUBLISHERS
        self.command_publisher = rospy.Publisher('/drone_command', PlutoMsg, queue_size=0)
        self.error_publishers = []
        for axis in self.sense_axes:
            self.error_publishers.append(rospy.Publisher('/%s_error' % axis, Float64, queue_size=1))

        # SUBSCRIBERS
        rospy.Subscriber('/whycon/poses', PoseArray, self._whycon_callback)
        rospy.Subscriber('/drone_yaw', Float64, self._yaw_callback)

        for index, axis in enumerate(self.sense_axes):
            rospy.Subscriber(
                '/pid_tuning_%s' %
                axis,
                PidTune,
                lambda msg: self._set_pid_callback(
                    msg,
                    index))
        self.arm()
        print("Initialized Drone")

    def disarm(self):
        """
        Disarm the Drone.
        Drone should be almost landing as this shuts down engines suddenly.
        Use land() if soft landing is required.
        """
        rospy.sleep(1)
        # aux4 is used to arm and disarm the drone
        self.publish_command(aux4=1000)
        rospy.sleep(1)
        print('Disarmed')

    def arm(self):
        """
        Arm the drone.
        Ready for takeoff with a low starting throttle.
        """
        self.disarm()
        rospy.sleep(1)
        self.publish_command(aux4=1500, throttle=1000)
        rospy.sleep(1)
        print('Armed')

    def land(self):
        """
        Land the Drone.
        Stabilize the drone and then slowly decrease throttle
        """
        self.publish_command()
        for throttle in range(1500, 1400, -10):
            self.publish_command(throttle=throttle)
            rospy.sleep(.5)
        self.disarm()

    def _yaw_callback(self, msg):
        """
        Callback for orientation
        """
        if 'yaw' in self.cartesian_axes:
            self.drone_position[self.cartesian_axes.index('yaw')] = msg.data
        else:
            self.drone_position[self.cartesian_axes.index('yaw*')] = - msg.data

    def _whycon_callback(self, msg):
        """
        Callback for coordinates
        """
        for axis in ('x', 'y', 'z'):
            if axis in self.cartesian_axes:
                self.drone_position[self.cartesian_axes.index(
                    axis)] = getattr(msg.poses[0].position, axis)
            else:
                self.drone_position[self.cartesian_axes.index(
                    axis + '*')] = - getattr(msg.poses[0].position, axis)
        # Experimental Constant for scaling
        self.drone_position[:2] /= 7.55
        # Experimental Constant for scaling
        self.drone_position[2] = -self.drone_position[2] * .0549 + 3.037

    def _set_pid_callback(self, msg, index):
        """
        Callback for setting pid externally
        """
        self.Kp[index] = msg.Kp * 0.06
        self.Ki[index] = msg.Ki * 0.008
        self.Kd[index] = msg.Kd * 0.3

    def publish_command(
            self,
            pitch=1500,
            roll=1500,
            throttle=1500,
            yaw=1500,
            aux4=1500):
        """
        Send a command to the drone.
        All the arguments are optional, if called without any argument, it stabilizes drone to neutral position.
        """
        self.cmd_msg.rcThrottle = throttle
        self.cmd_msg.rcRoll = roll
        self.cmd_msg.rcPitch = pitch
        self.cmd_msg.rcYaw = yaw
        self.cmd_msg.rcAUX4 = aux4
        self.command_publisher.publish(self.cmd_msg)

    def publish_error(self):
        """
        Publish current error values to respective topics
        """
        for index, axis in enumerate(self.sense_axes):
            self.error_msg.data = self.error[index]
            self.error_publishers[index].publish(self.error_msg)

    def _calculate_response(self):
        """
        PID algorithm
        """
        dt = self.current_time - self.previous_time
        de = self.error - self.previous_error
        self.previous_error = self.error
        # Condition for integral coefficient to remove reset integral windup
        self.cumulative_error = np.where((self.min_Ki_margin < abs(self.error)) & (
                abs(self.error) < self.max_Ki_margin), self.cumulative_error + self.error * dt, 0)
        response = self.Kp * self.error + self.Ki * self.cumulative_error + self.Kd * (de / dt)
        response += self.base_values
        # Clip to maximum and minimum values
        return np.clip(response, self.min_values, self.max_values)

    def pid(self):
        """
        PID Controller implementation
        """
        self.current_time = time.time()
        if self.sample_time < self.current_time - self.previous_time:
            self.error = self.setpoint - self.drone_position
            self.publish_error()
            response = self._calculate_response()
            self.publish_command(**dict(zip(self.control_axes, response)))
            self.previous_time = self.current_time

    def reach_target(self, setpoint):
        """
        Home in to a particular target or setpoint
        setpoint is supplied as a pose numpy array
        """
        self.setpoint = setpoint
        while not rospy.is_shutdown():
            self.pid()


if __name__ == '__main__':
    """
    Task 1.1
    """
    e_drone = Edrone()
    e_drone.reach_target(np.array([1.1094, -.6597, +1.5194, 0.00]))
