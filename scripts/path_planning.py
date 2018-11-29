#!/usr/bin/env python
"""
Submission: Hungry Bird Team 2334

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

from __future__ import print_function, division, with_statement
from itertools import count
import time

import numpy as np
import rospy
from enum import Enum
from geometry_msgs.msg import PoseArray, Pose
from pid_tune.msg import PidTune
from plutodrone.msg import PlutoMsg
from std_msgs.msg import Float64

# set global printing options for numpy
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class Edrone:
    """
    Class corresponding to the E-Drone
    It is used to control the simulated or real E-Drone using an inbuilt pid controller.
    Commands are sent to the drone which continues following one command till the next is sent.

    Regarding Coordinate Systems:
    Internally the coordinates or poses of the drone are represented as numpy arrays in the
    order given by the sense_axes and control_axes variable. This ordering does not change.
    The order of the input coordinates is dictated by the cartesian_axes variable which may be changed
    if there is a mismatch in the physical orientation of coordinate systems.
    Currently 'pitch' corresponds to x-axis and so on.

    Usage:

    To initialize Drone:
    >>> my_drone = Edrone()     # First disarmed for safety reasons
    Disarmed
    Armed
    Initialized Drone

    To publish a command to the Drone:
    >>> my_drone.publish_command(throttle = 1550) #base value is 1500
    <Publisher Output>

    >>> my_drone.publish_command(roll=1550)  #base value is 1500
    <Publisher Output>

    >>> my_drone.publish_command() # Return the drone to equilibrium
    <Publisher Output>

    >>> target_pose = np.array([1,4,6,0]) # ['pitch', 'roll', 'altitude', 'yaw']
    >>> my_drone.reach_target(target_pose) # Hover Near A target
    <Publisher outputs>

    >>> my_drone.disarm()  # Disarm Drone
    Disarmed

    >>> my_drone.arm() # Ready For Action
    Armed

    >>> my_drone.land()
    Disarmed
    """

    # AXES AND ORIENTATIONS
    # Ordering of values in numpy arrays used internally to represent pose.This order should not be changed.
    sense_axes = Enum('sense_axes', zip('pitch roll altitude yaw'.split(),count()))
    control_axes = Enum('control_axes', zip('pitch roll throttle yaw'.split(),count()))
    # You can change the order here if there is a mismatch in coordinate
    cartesian_axes = Enum('cartesian_axes', zip('x y z'.split(),count()))

    # CONTROLLER CONSTANTS
    # To be read as ['pitch', 'roll', 'altitude', 'yaw']
    Kp = np.array([30, 30, 40, 0])
    Ki = np.array([70, 70, 50, 0])
    Kd = np.array([0, 0, 350, 0])
    max_Ki_margin = np.array([.05, .05, .04, 0.0])
    min_Ki_margin = np.array([0.02, 0.02, .02, 0.0])
    max_values = np.array([1515, 1515, 1800, 1800])
    min_values = np.array([1485, 1485, 1200, 1200])
    base_values = np.array([1500, 1500, 1500, 1500])
    sample_time = 0.10

    # TRANSFORMATION CONSTANTS
    scaling_slope = np.array([-0.13245, -0.13245, -.0549, 1.000])
    scaling_intercept = np.array([0, 0, 3.037, 0])

    def __init__(self):
        """
        Initialize the drone
        Automatically arm and bring the drone to a usable state
        Also initialize all the subscriptions and publications of the node
        Some initial values for the PID controller are defined here
        """
        rospy.init_node('drone_control')

        # STATE VARIABLES
        self.setpoint = np.zeros(4)
        self.path = []
        self.drone_position = np.zeros(4)
        self.current_time = time.time()
        self.previous_time = time.time()
        self.previous_error = np.zeros(4)
        self.cumulative_error = np.zeros(4)
        self.error = np.zeros(4)
        self.path_received = False

        # I/O INITIALIZATION
        self.target_msg = Pose()
        self.cmd_msg = PlutoMsg()
        self.error_msg = Pose()

        # PUBLISHERS
        self.command_publisher = rospy.Publisher('/drone_command', PlutoMsg, queue_size=1)
        self.target_publisher = rospy.Publisher('/path_target', Pose, queue_size=1)
        self.error_publisher = rospy.Publisher('/drone_error', Pose, queue_size=1)

        # SUBSCRIBERS
        rospy.Subscriber('/whycon/poses', PoseArray, self._whycon_callback)
        rospy.Subscriber('/drone_yaw', Float64, self._yaw_callback)
        rospy.Subscriber('/vrep/waypoints', PoseArray, self._path_callback)

        for axis in self.sense_axes:
            rospy.Subscriber('/pid_tuning_%s' % axis.name, PidTune, lambda msg: self._set_pid_callback(msg, axis.value))
        self.arm()

        print("Initialized Drone")

    # ACTION VERBS

    def disarm(self):
        """
        Disarm the Drone.
        Drone should be almost landing as this shuts down engines abruptly.
        Use land() if soft landing is required.
        """
        rospy.sleep(1)
        self.publish_command(aux4=1000)
        rospy.sleep(1)

        print('Disarmed')

    def arm(self):
        """
        Arm the drone.
        Drone is armed when aux4 is greater than a set value (Defined in lua code)
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

    def is_reached(self,error_tolerance):
        return np.all(abs(self.error) < error_tolerance)

    def set_target(self, target):
        self.setpoint = target
        self.error = self.setpoint - self.drone_position

    def reach_target(self, target , tolerance=[.0823, .0823, .0823, 1000]):
        """
        Home in to a particular target or setpoint
        setpoint is supplied as a pose numpy array
        """
        self.set_target(target)
        while not rospy.is_shutdown() and not self.is_reached(tolerance):
            self._pid()
        self.publish_command(aux4=1320)

    def get_path(self,target):
        self.path = []
        self.path_received = False
        self._to_pose_from_array(self.target_msg, target)
        self.target_publisher.publish(self.target_msg)
        while not rospy.is_shutdown() and not self.path_received:
            rospy.sleep(.1)  # Wait for path
        print(len(self.path))

    def reach_target_via_path(self, target):
        """
        Navigate to a target via path planning using Vrep OMPL Plugin
        First send a command to Vrep to plan the path
        Wait for the path. After getting path, reach each point in the path.
        """
        self.get_path(target)

        
        for pose in self.path[::2]:
            #print(pose)
            #print('p', self.drone_position)
            #print('e', self.error)
            self.reach_target(pose,tolerance = [.1023, .10623, .10623, 100])
            self.publish_command()


    # PUBLISHING VERBS

    def publish_command(self, pitch=1500, roll=1500, throttle=1500, yaw=1500, aux4=1500):
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
        self._to_pose_from_array(self.error_msg, self.error)
        self.error_publisher.publish(self.error_msg)

    # CALLBACKS

    def _yaw_callback(self, msg):
        """
        Callback for orientation
        """
        self.drone_position[3] = msg.data

    def _whycon_callback(self, msg):
        """
        Callback for coordinates
        """

        self._from_pose_to_array(msg.poses[0], self.drone_position)
        self.drone_position = self.drone_position * self.scaling_slope + self.scaling_intercept

    def _set_pid_callback(self, msg, index):
        """
        Callback for setting pid externally
        """
        self.Kp[index] = msg.Kp * 0.06
        self.Ki[index] = msg.Ki * 0.008
        self.Kd[index] = msg.Kd * 0.3

    def _path_callback(self, msg):
        self.path = []
        for pose in msg.poses:
            array = np.zeros(4)
            self._from_pose_to_array(pose, array)
            self.path.append(array)
        self.path_received = True

    # UTILITY FUNCTIONS

    def _to_pose_from_array(self, pose, array):
        for axis in self.cartesian_axes:
            setattr(pose.position, axis.name, array[axis.value])

    def _from_pose_to_array(self, pose, array):
        for axis in self.cartesian_axes:
            array[axis.value] = getattr(pose.position, axis.name)

    def _calculate_response(self):
        """
        PID algorithm response calcultion
        """
        dt = self.current_time - self.previous_time
        de = self.error - self.previous_error
        self.previous_error = self.error

        # Condition for integral coefficient to remove reset integral windup
        self.cumulative_error = np.where((self.min_Ki_margin < abs(self.error)) & (
            abs(self.error) < self.max_Ki_margin), self.cumulative_error + self.error * dt, 0)

        # PID Calculation
        response = self.Kp * self.error + self.Ki * self.cumulative_error + self.Kd * (de / dt)

        # Rotation to the drone frame
        yaw = np.radians(self.error[3])
        c, s = np.cos(yaw), np.sin(yaw)
        print('assa',response)
        x,y=response[:2]
        response[0] = c*x-s*y
        response[1] =s*x+c*y
        print ('as',response)

        # Add Base values and Clip to maximum and minimum values
        response += self.base_values
        return np.clip(response, self.min_values, self.max_values)

    def _pid(self):
        """
        PID Controller implementation
        """
        self.current_time = time.time()
        if self.sample_time < self.current_time - self.previous_time:
            self.error = self.setpoint - self.drone_position
            print('e', self.error)
            print('p', self.drone_position)
            self.publish_error()
            response = self._calculate_response()
            self.publish_command(**{axis.name: response[axis.value] for axis in self.control_axes})
            self.previous_time = self.current_time


if __name__ == '__main__':
    """
       Task 1.2:
       Implementing Path Planning to Avoid Obstacles.
       Visit all goal points while evading the obstacles.
       Implement path planning through OMPL Plugin For Vrep
    """
    e_drone = Edrone()
    goals = (
        np.array([-.75, .25, 1.225, 0]),  # Initial Waypoint
        np.array([1.75, 0.075, 0.75, 0]),  # Goal 1
        np.array([-.75, -1.1, 0.675, 0]),  # Goal 2
    )
    #e_drone.publish_command(yaw=2000)
    #time.sleep(6)
    
    #e_drone.get_path(goals[0])
    e_drone.reach_target(goals[0])
    
    #e_drone.get_path(goals[1])
    e_drone.reach_target(np.array([.5,-.20,1,0]),tolerance=[.1023, .1623, .1023, 1000])
    e_drone.reach_target(goals[1])
    
    #e_drone.get_path(goals[2])
    e_drone.reach_target(goals[2])

    #e_drone.get_path(goals[0])
    e_drone.reach_target(goals[0])
    #for goal in goals:
 #       e_drone.reach_target(goal)
  #      e_drone.publish_error()
    # e_drone.publish_error()
    e_drone.disarm()
    
