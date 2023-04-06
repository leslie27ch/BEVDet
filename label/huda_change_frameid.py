
#! /usr/bin/python3

from multiprocessing.sharedctypes import Value
import sys
import os
# import rclpy
from typing import Union
import numpy as np
# import rosbag2_py
#from sensor_msgs_py.point_cloud2 import read_points
import sensor_msgs
import tf2_msgs
# import delphi_esr_msgs
import nav_msgs
from secrets import token_hex
import json
from datetime import datetime
from tqdm import tqdm
import itertools
import yaml
# from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message
import cv2
from cv_bridge import CvBridge

import math
import rosbag
import sensor_msgs.point_cloud2 as pc2
import os
file_root = "/home/qiu/huda_bag_frameid"
bag_data = rosbag.Bag("/home/qiu/bagdir2/2023-01-07-13-23-38_3.bag")
# perception_data = bag_data.read_messages(topics = ["/n_camera_gmsl_prep"])
# for k,(topic, msg, timestamp) in enumerate(perception_data):
#     print(topic,timestamp)
#     img_converter = CvBridge()
#     for i,msg_i in enumerate(msg.raw_image):
#
#         filename = file_root + "_" +str(i)
#         if not os.path.isdir(filename):
#             os.mkdir(filename)
#         filename = filename+"/"+ str(k) + ".jpg"
#         img = cv2.cvtColor(img_converter.imgmsg_to_cv2(msg_i), cv2.COLOR_BGR2RGB)
#         print(i,msg_i.header.stamp,img.shape[0],img.shape[1])
#         cv2.imwrite(filename, img)

perception_data = bag_data.read_messages(topics = ["/n_camera_gmsl_prep","/iv_points","/os_cloud_node_1/points","/os_cloud_node_2/points","/rslidar_points"])
new_bag = "/2023-01-07-13-23-38_3.bag"
new_bag = file_root + new_bag
outbag = rosbag.Bag(new_bag, 'w')
for k,(topic, msg, timestamp) in enumerate(perception_data):
    if topic == "/n_camera_gmsl_prep":
        msg.header.stamp = msg.raw_image[0].header.stamp

        msg.header.frame_id = "n_camera_gmsl_prep"

    elif topic =="/iv_points":
        msg.header.frame_id = "lidar_perception_front"
    elif topic =="/os_cloud_node_1/points":
        msg.header.frame_id = "lidar_perception_left"
    elif topic =="/os_cloud_node_2/points":
        msg.header.frame_id = "lidar_perception_right"
    elif topic =="/rslidar_points":
        msg.header.frame_id = "lidar_perception_back"
    else:
        print("error:other topic exist ")
        break
    outbag.write(topic, msg, msg.header.stamp)

outbag.close()
outbag.reindex()

