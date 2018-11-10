#!/usr/bin/env pytho
"""
Submission: Hungry Bird Team 2334

This python file runs a ROS-node of name drone_control which holds the position of e-Drone on the given dummy.

This node publishes and subsribes the following topics:
        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command          /whycon/poses
        /alt_error              /pid_tuning_altitude
        /pitch_error            /pid_tuning_pitch
        /roll_error             /pid_tuning_roll
        /yaw_error              /pid_tuning_yaw
                                /drone_yaw
"""

from __future__ import print_function, division

import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray
from pid_tune.msg import PidTune
from plutodrone.msg import *
from std_msgs.msg import Float64
from std_msgs.msg import Int16
from std_msgs.msg import Int64


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

    >>> my_drone.publish_command(throttle = 1550)
    <Publisher Output>

    >>> my_drone.publish_command(roll=1550)
    <Publisher Output>

    >>> my_drone.publish_command() # Return the drone to equilibrium
    <Publisher Output>

    >>> my_drone.reach_target(target_pose) # Hover Near A target
    <Publisher outputs>

    >>> my_drone.disarm()  # Disarm Drone
    Disarmed

    >>> my_drone.arm() # Ready For Action
    Armed


    Regarding Coordinate Systems:
    Internally the coordinates or poses of the drone are represented as numpy arrays in the order given by the
    sense_axes and control_axes variable. This ordering does not change.
    The order of the coordinates is dictated by the cartesian_axes variable which may be changed if there is a mismatch
    in the physical orientation of coordinate systems. Currently 'pitch' corresponds to x-axis and so on.

    """

    # Ordering of values in numpy arrays used internally to represent pose.These orders should not be changed as
    # it is the order of  values in the numpy array, which are hardcoded
    sense_axes = ('pitch', 'roll', 'altitude', 'yaw')
    control_axes = ('pitch', 'roll', 'throttle', 'yaw')

    # You can change the order here if there is a mismatch in coordinate systems. Use x* etc to give negative axis
    cartesian_axes = ('x', 'y', 'z', 'yaw')

    def __init__(self):
        """
        Initialize the drone
        Automatically arm and bring the drone to a usable state
        Also initialize all the subscriptions and publications of the node
        Some initial values for the PID controller are defined here
        """
        rospy.init_node('drone_control')
        self.setpoint = np.zeros(4)
        self.drone_position = np.zeros(4)

        self.cmd = PlutoMsg()

        self.sample_time = 0.060
        self.Kp = np.array([0.075, 0.075, 0, 1.0])
        self.Ki = np.array([0.075, 0.075, 0, 0])
        self.Kd = np.array([0.0125, 0.0125, 0, 0])
        self.max_values = np.array([1800, 1800, 1800, 1800])
        self.min_values = np.array([1200, 1200, 1200, 1200])
        self.base_values = np.array([1500, 1500, 1500, 1500])

        self.current_time = time.time()
        self.previous_time = time.time()
        self.previous_error = np.zeros(4)
        self.cumulative_error = np.zeros(4)

        self.command_publisher = rospy.Publisher('/drone_command', PlutoMsg, queue_size=0)
        self.error_publishers = []
        for axis in self.sense_axes:
            self.error_publishers.append(rospy.Publisher('/%s_error' % axis, Float64, queue_size=1))

        rospy.Subscriber('/whycon/poses', PoseArray, self._whycon_callback)
        rospy.Subscriber('/drone_yaw', PoseArray, self._yaw_callback)

        for index, axis in enumerate(self.sense_axes):
            rospy.Subscriber('/pid_tuning_%s' % axis, PidTune, lambda msg: self._set_pid(msg, index))
        self.arm()
        print("Initialized Drone")

    def disarm(self):
        """
        Disarm the Drone.
        Drone should be almost landing as this shuts down engines suddenly.
        Use land() if soft landing is required.
        """
        rospy.sleep(1)
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
        for throttle in range(1500, 1100, -100):
            self.publish_command(throttle=throttle)
            rospy.sleep(.5)
        self.disarm()

    def _yaw_callback(self, msg):
        if 'yaw' in self.cartesian_axes:
            self.drone_position[self.cartesian_axes.index('yaw')] = msg.data
        else:
            self.drone_position[self.cartesian_axes.index('yaw*')] = - msg.data

    def _whycon_callback(self, msg):
        for axis in ('x', 'y', 'z'):
            if axis in self.carteswhycon_callbackian_axes:
                self.drone_position[self.cartesian_axes.index(axis)] = getattr(msg.poses[0].position, axis)
            else:
                self.drone_position[self.cartesian_axes.index(axis + '*')] = - getattr(msg.poses[0].position, axis)

    def _set_pid(self, msg, index):
        self.Kp[index] = msg.Kp * 0.06
        self.Ki[index] = msg.Ki * 0.008
        self.Kd[index] = msg.Kd * 0.3

    def publish_command(self, pitch=1500, roll=1500, throttle=1500, yaw=1500, aux4=1500):
        """
        Send a command to the drone.
        All the arguments are optional, if called without any argument, it stabilizes drone to neutral position.
         
        """
        self.cmd.rcThrottle = throttle
        self.cmd.rcRoll = roll
        self.cmd.rcPitch = pitch
        self.cmd.rcYaw = yaw
        self.cmd.rcAUX4 = aux4
        self.command_publisher.publish(self.cmd)
        print(self.cmd)

    def publish_error(self, error):
        msg = Float64()
        for index, axis in enumerate(self.sense_axes):
            msg.data = error[index]
            self.error_publishers[index].publish(msg)
        print(error)

    def _calculate_response(self, error):
        dt = self.current_time - self.previous_time
        de = error - self.previous_error
        self.cumulative_error += error * dt
        response = self.Kp * error + self.Ki * self.cumulative_error + self.Kd * de / dt
        response += self.base_values
        return np.clip(response, self.min_values, self.max_values)

    def pid(self):
        """
        PID Controller implementation
        """
        self.current_time = time.time()
        if self.sample_time > self.current_time - self.previous_time:
            error = self.setpoint - self.drone_position
            self.publish_error(error)
            response = self._calculate_response(error)
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
