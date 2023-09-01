#!/usr/bin/env python3

from ast import Pass
import rospy 
from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud

rospy.init_node("scan_obstacles")
rospy.wait_for_service("assemble_scans")

assemble_scans = rospy.ServiceProxy('assemble_scans', AssembleScans)
pub = rospy.Publisher ("/pointcloud", PointCloud, queue_size=1)
r = rospy.Rate (10)

while not rospy.is_shutdown():

    ###assemble_scans(begin_time, time_end)
    try:
        resp = assemble_scans(rospy.Time(0,0), rospy.get_rostime())
        pub.publish (resp.cloud)
        r.sleep()
    except:
        rospy.loginfo("Pointcloud was not available!")
        Pass