#!/usr/bin/env python
from __future__ import print_function

import rospy
import roslib
import tf

from geometry_msgs.msg import PoseArray
from aruco_msgs.msg import MarkerArray


#Defining a class
class Marker_detect():

	def __init__(self):
		rospy.init_node('marker_detection',anonymous=False) # initializing a ros node with name marker_detection

		self.whycon_marker = {}	# Declaring dictionaries
		self.aruco_marker = {}

		rospy.Subscriber('/whycon/poses',PoseArray,self.whycon_data)	# Subscribing to topic
		rospy.Subscriber('/aruco_marker_publisher/markers',MarkerArray,self.aruco_data)	# Subscribing to topic
		


	# Callback for /whycon/poses
	def whycon_data(self,msg):
                for index,marker in enumerate(msg.poses):
                        self.whycon_marker[index]=[marker.position.x,marker.position.y,marker.position.z]


	# Callback for /aruco_marker_publisher/markers
	def aruco_data(self,msg):
                for marker in msg.markers:
                       self.aruco_marker[marker.id]=[marker.pose.pose.orientation.x,marker.pose.pose.orientation.y,marker.pose.pose.orientation.z,marker.pose.pose.orientation.w]
		


		# Printing the detected markers on terminal
		print ("\nWhyCon_marker",self.whycon_marker)
		print ("ArUco_marker",self.aruco_marker)




if __name__=="__main__":

	marker = Marker_detect()

	
	while not rospy.is_shutdown():
		rospy.spin()
