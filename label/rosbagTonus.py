
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




def write_scene(argdict, use_hz = 0.49, key_sensor = 'hello',total_num_lidar=4):

    bag_dir = os.path.normpath(os.path.abspath(argdict["bag_in"]))
    rosbag_file = argdict["bag_in"]
    #metadatafile: str = os.path.join(bag_dir, "metadata.yaml")
    param_file = os.path.normpath(os.path.abspath(argdict["param_file"]))
    # Read in param file
    if not os.path.isfile(param_file):
        raise ValueError("Param file %s does not exist" % param_file)
    with open(param_file, "r") as f:
        param_dict: dict = yaml.load(f, Loader=yaml.SafeLoader)
    # Extract info from param file
    track_name = param_dict["BAG_INFO"]["TRACK"]
    map_name = param_dict["BAG_INFO"]["MAP"]
    lidar_topics = dict()
    radar_topics = dict()
    camera_topics = dict()
    camera_calibs = dict()

    for sensor_name in param_dict["SENSOR_INFO"]:
        modality = sensor_name.split('_')[0]
        if modality == "LIDAR":
            if param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]:
                lidar_topics[param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]] = sensor_name
        elif modality == "RADAR":
            if param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]:
                radar_topics[param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]] = sensor_name
        elif modality == "CAMERA":
            if param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]:
                camera_topics[param_dict["SENSOR_INFO"][sensor_name]["TOPIC"]] = sensor_name
                camera_calibs[param_dict["SENSOR_INFO"][sensor_name]["CALIB"]] = []
        else:
            raise ValueError(
                "Invalid sensor %s in %s. Ensure sensor is of type LIDAR, RADAR, or CAMERA and is named [SENSOR TYPE]_[SENSOR LOCATION]" % (
                sensor_name, param_file))
    bag_data = rosbag.Bag(rosbag_file, 'r')
    info_dict = yaml.load(bag_data._get_yaml_info(), Loader=yaml.SafeLoader)

    # Extract metadata from bag directory
    # if not os.path.isfile(metadatafile):
    #     raise ValueError(
    #         "Metadata file %s does not exist. Are you sure %s is a valid rosbag?" % (metadatafile, bag_dir))
    # with open(metadatafile, "r") as f:
    #     metadata_dict: dict = yaml.load(f, Loader=yaml.SafeLoader)["rosbag2_bagfile_information"]
    # topic_types, type_map, topic_metadata_map, reader = open_bagfile(bag_dir)
    # topic_count_dict = {entry["topic_metadata"]["name"]: entry["message_count"] for entry in
    #                     metadata_dict["topics_with_message_count"]}
    # topic_counts = np.array(list(topic_count_dict.values()))
    # total_msgs = np.sum(topic_counts)

    msg_dict  = {entry["topic"]: [] for entry in info_dict["topics"]}

    #from now on, meta data and param be stored in msg_dict and param_dict

    perception_data = bag_data.read_messages()
    if 'ODOM_TOPIC' in param_dict["BAG_INFO"]:
        for topic, msg, t in perception_data:
            if topic == param_dict["BAG_INFO"][
                    "ODOM_TOPIC"] :
                    # msg_type = type_map[topic]
                    # msg_type_full = get_message(msg_type)
                    # msg = deserialize_message(data, msg_type_full)
                    msg_dict[topic].append((t, msg))

    #-------------------------------------------------------------------------------------------------

    # make a big rosbag with hous into little rosbag with one minute
    # for example , big rosbag :n015-2018-07-24-11-22-45+0800 , little rosbag : n015-2018-07-24-11-22-45+0800_1  n015-2018-07-24-11-22-45+0800_2 .....
    # little rosbag  is a scene
    # big rosbag is a log

    # Create log.json, in one day is a log and is
    #rosbag: 2023-01-02_i.bag
    print("creating or adding log.json")
    log_token = token_hex(16)
    if not os.path.exists('v1.0-mini'):
        os.mkdir('v1.0-mini')
    log_token = token_hex(16)
    if os.path.exists('v1.0-mini/log.json'):
        with open('v1.0-mini/log.json', 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    log_exist = False
    log_file = rosbag_file.split('/')[-1].split('_')[0]
    for entry in logs:
        if log_file == entry['logfile']:
            log_exist = True
    if log_exist == False:
        new_log = dict()

        new_log['token'] = log_token
        if rosbag_file[-1] == '/':
            rosbag_file = rosbag_file[:-1]
        new_log['logfile'] = log_file
        new_log['vehicle'] = param_dict["BAG_INFO"]["TEAM"]
        date_captured = datetime.fromtimestamp(bag_data.get_start_time()).strftime('%Y-%m-%d')
        new_log['date_captured'] = date_captured
        new_log['location'] = track_name
        logs.append(new_log)
    with open('v1.0-mini/log.json', 'w') as f:
        json.dump(logs, f, indent=4)


    # Create map.json
    print("creating or adding map.json")
    # If map file already exists, add log_token to existing map
    if os.path.exists('v1.0-mini/map.json'):
        with open('v1.0-mini/map.json', 'r') as f:
            maps = json.load(f)
        new_json = []
        map_exists = False
        for map in maps:
            if map_name == map["name"]:
                map_exists = True
            if map_exists == True and log_exist == False:
                map['log_tokens'].append(log_token)

            new_json.append(map)

        if not map_exists:
            new_map = dict()
            new_map['token'] = token_hex(16)
            new_map['log_tokens'] = [log_token]
            new_map['category'] = "semantic_prior"
            new_map['filename'] = 'map/'+new_map['token'] +'.png'
            new_json.append(new_map)
        with open('v1.0-mini/map.json', 'w') as f:
            json.dump(new_json, f, indent=4)
    else:
        # Case where map.json does not exist
        with open('v1.0-mini/map.json', 'w') as f:
            map = dict()
            map['token'] = token_hex(16)
            map['log_tokens'] = [log_token]
            map['category'] = "semantic_prior"
            map['name'] = map_name
            map['filename'] = 'map/'+ map['token'] +'.png'
            json.dump([map], f, indent=4)

    # # Create sensor.json
    print("creating or adding sensor.json")
    sensor_token_dict = dict()
    if not os.path.exists('v1.0-mini/sensor.json'):
        sensor_configs = []
        for channel in param_dict['SENSOR_INFO']:
            sensor_config = dict()
            sensor_token = token_hex(16)

            # Store sensor token for use in calibrated_sensor.json
            sensor_token_dict[param_dict['SENSOR_INFO'][channel]['CALIB']] = sensor_token

            sensor_config['token'] = sensor_token
            sensor_config['channel'] = channel
            sensor_config['modality'] = channel.split('_')[0].lower()
            os.makedirs('samples/%s' % channel)
            os.makedirs('sweeps/%s' % channel)
            sensor_configs.append(sensor_config)

        with open('v1.0-mini/sensor.json', 'w') as f:
            json.dump(sensor_configs, f, indent=4)
    # If sensor.json exists, load existing tokens to param_dict
    else:
        with open('v1.0-mini/sensor.json', 'r') as f:
            sensors = json.load(f)
            for sensor in sensors:
                sensor_token_dict[param_dict['SENSOR_INFO'][sensor['channel']]['CALIB']] = sensor['token']

    # # Create calibrated_sensor.json
    print("creating or adding calibrated_sensor.json")

    if os.path.exists('v1.0-mini/calibrated_sensor.json'):
        with open('v1.0-mini/calibrated_sensor.json', 'r') as f:
            calibrated_sensors = json.load(f)
    else:
        calibrated_sensors = []
    frames_received = []

    # put biaoding info into calibrated_sensor.json
    with open(argdict["calibration"], 'r') as f:
        calibrated_dict = json.load(f)
    for key,entrys in calibrated_dict.items():
        for entry in entrys:
            if entry['name']in sensor_token_dict.keys() and entry['name'] not in frames_received:
                calibrated_sensor_data = dict()
                calibrated_sensor_token = token_hex(16)
                calibrated_sensor_data['token'] = calibrated_sensor_token
                calibrated_sensor_data['sensor_token'] = sensor_token_dict[entry['name']]
                calibrated_sensor_data['translation'] = entry['poses']['T']
                calibrated_sensor_data['rotation'] = entry['poses']['R']
                if entry['name']in camera_calibs:
                    calibrated_sensor_data['camera_intrinsic'] = entry['K']

                else:
                    calibrated_sensor_data['camera_intrinsic'] = []

                    # Store calibrated sensor token for use in creating sample_data.json
                sensor_token_dict[entry['name']] = calibrated_sensor_token
                # Mark frame as processed
                frames_received.append(entry['name'])
                calibrated_sensors.append(calibrated_sensor_data)
            if len(frames_received) == len(lidar_topics) + len(radar_topics) + len(camera_topics):
                break
    with open('v1.0-mini/calibrated_sensor.json', 'w') as f:
        json.dump(calibrated_sensors, f, indent=4)


    # # Create remaining json files
    if os.path.exists('v1.0-mini/sample.json'):
        with open('v1.0-mini/sample.json', 'r') as f:
            samples = json.load(f)
    else:
        samples = []
    if os.path.exists('v1.0-mini/sample_data.json'):
        with open('v1.0-mini/sample_data.json', 'r') as f:
            sample_data = json.load(f)
    else:
        sample_data = []
    if os.path.exists('v1.0-mini/ego_pose.json'):
        with open('v1.0-mini/ego_pose.json', 'r') as f:
            ego_poses = json.load(f)
    else:
        ego_poses = []
    ego_pose_queue = []

    print("Extracting odometry data")
    if 'ODOM_TOPIC' in param_dict["BAG_INFO"]:
        for timestamp, msg in msg_dict[param_dict["BAG_INFO"]["ODOM_TOPIC"]]:
            # Create ego_pose.json
            ego_pose = dict()
            ego_pose_token = token_hex(16)
            ego_pose_queue.append((ego_pose_token, timestamp, np.array(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])))
            ego_pose['token'] = ego_pose_token
            ego_pose['timestamp'] = timestamp
            ego_pose['rotation'] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                                    msg.pose.pose.orientation.w]
            ego_pose['translation'] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            ego_poses.append(ego_pose)



    print("Extracting sensor data")
    previous_sampled_timestamp = None
    first_sample = ''
    prev_sample_token, next_sample_token = '', token_hex(16)
    scene_token = token_hex(16)

    nbr_samples = 0
    previous_loc = 0
    data_token_dict = dict()
    img_converter = CvBridge()
    for topic in list(lidar_topics.keys()) + list(radar_topics.keys()) + list(camera_topics.keys()):
        data_token_dict[topic] = ['', token_hex(16)]

    sensors_added = set()
    first_meet_keysensor = False
    num_lidar = 0
    for topic, msg, timestamp in perception_data:
        if topic in camera_topics:
            sensor_name = camera_topics[topic]
            for msg_i in msg.raw_image:
                img = cv2.cvtColor(img_converter.imgmsg_to_cv2(msg_i), cv2.COLOR_BGR2RGB)
        if not (topic in lidar_topics or topic in radar_topics or topic in camera_topics):
            continue
        if topic in lidar_topics and first_meet_keysensor == False:
            num_lidar += 1
            continue
        # meet the last lidar
        if num_lidar ==4 and first_meet_keysensor == False:
            first_meet_keysensor = True
            num_lidar = 1
            prev_keyframe_timestamp = timestamp

            continue




        if not first_meet_keysensor:
            continue

        # Create sample_data.json
        sensor_token = sensor_token_dict[msg.header.frame_id]
        prev_data_token = data_token_dict[topic][0]
        data_token = data_token_dict[topic][1]
        # token pass on
        data_token_dict[topic][1] = token_hex(16)
        data_token_dict[topic][0] = data_token
        sensor_data = dict()
        sensor_data['token'] = data_token
        sensor_data['sample_token'] = next_sample_token
        sensor_data['calibrated_sensor_token'] = sensor_token
        # closest sample is the last sample
        # if msg.header.frame_id not in sensors_sample_info_added:
        #
        #     samples[-1]['data'][msg.header.frame_id] = data_token
        #
        #     sensors_sample_info_added.add(msg.header.frame_id)


        # Find closest ego pose
        previous_time_difference = abs(np.inf)
        for i in range(previous_loc, len(ego_pose_queue)):
            time_difference = abs((datetime.fromtimestamp(timestamp * 1e-9) - datetime.fromtimestamp(
                ego_pose_queue[i][1] * 1e-9)).total_seconds())
            if time_difference < previous_time_difference:
                previous_time_difference = time_difference
            else:
                ego_pose_entry = ego_pose_queue[i - 1]
                # ego_pose_entry is tuplr
                sensor_data['ego_pose_token'] = ego_pose_entry[0]
                ego_velocity = (ego_pose_queue[i][2] - ego_pose_queue[i - 1][2]) / (
                            datetime.fromtimestamp(ego_pose_queue[i][1] * 1e-9) - datetime.fromtimestamp(
                        ego_pose_queue[i - 1][1] * 1e-9)).total_seconds()
                previous_loc = i
                break


        # Save data
        if topic in lidar_topics:
            sensor_name = lidar_topics[topic]
            height = 0
            width = 0
            saved_points = np.zeros((msg.width, 5))
            point_num = 0
            for point in pc2.read_points(msg, skip_nans=True):
                saved_points[point_num, 0] = point[0]
                saved_points[point_num, 1] = point[1]
                saved_points[point_num, 2] = point[2]
                saved_points[point_num, 3] = point[3]
                point_num += 1
            if sensor_name not in sensors_added:
                filename = "samples/{0}/{1}__{0}__{2}.pcd.bin".format(sensor_name, rosbag_file.split('/')[-1],
                                                                      timestamp)
                sensors_added.add(sensor_name)
                is_key_frame = True
                if first_sample == '':
                    first_sample = sample_token
            else:
                filename = "sweeps/{0}/{1}__{0}__{2}.pcd.bin".format(sensor_name, rosbag_file.split('/')[-1],
                                                                     timestamp)
                is_key_frame = False
            with open(filename, 'wb') as pcd_file:
                saved_points.astype('float32').tofile(pcd_file)
            sensor_data['fileformat'] = 'pcd'

        elif topic in camera_topics:
            sensor_name = camera_topics[topic]
            for msg_i in msg.raw_image:

                img = cv2.cvtColor(img_converter.imgmsg_to_cv2(msg_i), cv2.COLOR_BGR2RGB)
            height = img.shape[0]
            width = img.shape[1]
            if sensor_name not in sensors_added:
                filename = "samples/{0}/{1}__{0}__{2}.jpg".format(sensor_name, rosbag_file.split('/')[-1],
                                                                  timestamp)
                sensors_added.add(sensor_name)
                is_key_frame = True

            else:
                filename = "sweeps/{0}/{1}__{0}__{2}.jpg".format(sensor_name, rosbag_file.split('/')[-1], timestamp)
                is_key_frame = False
            cv2.imwrite(filename, img)
            sensor_data['fileformat'] = 'jpg'
        elif topic in radar_topics:
            sensor_name = radar_topics[topic]
            height = 0
            width = 0
            x_pos = msg.track_range * math.cos(math.radians(msg.track_angle))
            y_pos = msg.track_range * math.sin(math.radians(msg.track_angle))
            vx_comp = msg.track_range_rate
            vy_comp = msg.track_lat_rate
            vx = vx_comp + ego_velocity[0]
            vy = vy_comp + ego_velocity[1]
            dyn_prop = not (vx < 1.0 and vy < 1.0)
            ambig_state = 3 if dyn_prop else 4
            points = np.array(
                [x_pos, y_pos, 0.0, dyn_prop, msg.track_id, msg.track_width, vx, vy, vx_comp, vy_comp, 1,
                 ambig_state, int(1 / math.sqrt(2) * x_pos), int(1 / math.sqrt(2) * y_pos), 0, 1,
                 int(1 / math.sqrt(2) * vx), int(1 / math.sqrt(2) * vy)])
            if sensor_name not in sensors_added:
                filename = "samples/{0}/{1}__{0}__{2}.pcd".format(sensor_name, rosbag_file.split('/')[-1],
                                                                  timestamp)
                sensors_added.add(sensor_name)
                is_key_frame = True

            else:
                filename = "sweeps/{0}/{1}__{0}__{2}.pcd".format(sensor_name, rosbag_file.split('/')[-1], timestamp)
                is_key_frame = False
            with open(filename, 'wb') as pcd_file:
                pcd_file.write(
                    "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\nTYPE F F F I I F F F F F I I I I I I I I\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA binary\n".encode(
                        'utf-8'))
                points.tofile(pcd_file)
            sensor_data['fileformat'] = 'pcd'

        if topic in lidar_topics:
            equal_time = timestamp
            num_lidar += 1
            if num_lidar ==total_num_lidar and equal_time == timestamp:
                # the topic is the last lidar
                key_sensor = topic
                num_lidar = 1
            else:
                num_lidar = 1
                key_sensor="hello"

        # if topic is the last lidar
        if topic == key_sensor and (timestamp.secs + timestamp.nsecs * 1e-9) - (
                prev_keyframe_timestamp.secs + prev_keyframe_timestamp.nsecs * 1e-9) > use_hz:
            prev_keyframe_timestamp = timestamp
            nbr_samples += 1
            # Create sample.json
            sample = dict()
            sample_token = next_sample_token
            sample['token'] = sample_token
            sample['timestamp'] = timestamp.secs * 1e9 + timestamp.nsecs
            sample['scene_token'] = scene_token
            sample['prev'] = prev_sample_token
            prev_sample_token = next_sample_token
            next_sample_token = token_hex(16)
            sample['next'] = next_sample_token
            key_sensor ="hello"

            samples.append(sample)

            sensors_added.clear()

            if first_sample == '':
                first_sample = sample_token

        sensor_data['filename'] = filename


        sensor_data['is_key_frame'] = is_key_frame
        sensor_data['height'] = height
        sensor_data['width'] = width
        sensor_data['timestamp'] = timestamp.secs * 1e9 + timestamp.nsecs
        sensor_data['prev'] = prev_data_token
        sensor_data['next'] = data_token_dict[topic][1]
        sample_data.append(sensor_data)

    samples[-1]['next'] = ''
    sensors_cleared = set()
    for sensor_data in reversed(sample_data):
        if sensor_data['calibrated_sensor_token'] not in sensors_cleared:
            sensor_data['next'] = ''
            sensors_cleared.add(sensor_data['calibrated_sensor_token'])
        if len(sensors_cleared) == len(lidar_topics) + len(radar_topics) + len(camera_topics):
            break
    with open('v1.0-mini/sample.json', 'w') as f:
        json.dump(samples, f, indent=4)
    with open('v1.0-mini/sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=4)
    with open('v1.0-mini/ego_pose.json', 'w') as f:
        json.dump(ego_poses, f, indent=4)

    # Create scene.json
    if os.path.exists('v1.0-mini/scene.json'):
        with open('v1.0-mini/scene.json', 'r') as f:
            scenes = json.load(f)
    else:
        scenes = []
    with open('v1.0-mini/scene.json', 'w') as f:
        scene = dict()
        scene['token'] = scene_token
        scene['log_token'] = log_token
        scene['nbr_samples'] = nbr_samples
        scene['first_sample_token'] = first_sample
        scene['last_sample_token'] = sample_token
        if rosbag_file[-1] == '/':
            rosbag_file = rosbag_file[:-1]
        scene['name'] = rosbag_file.split('/')[-1].split('_')[0]
        scene['description'] = param_dict["BAG_INFO"]["DESCRIPTION"]
        scenes.append(scene)
        json.dump(scenes, f, indent=4)




def all_path(dirname):

    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        print("1:", maindir)  # 当前主目录
        print("2:", subdir)  # 当前主目录下的所有目录
        print("3:", file_name_list)  # 当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)

    return result

def rosbagStamp(bag):

    dir_path = "/home/qiu/bagdir2/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_name = bag.split("/")[-1]
    new_bag = os.path.join(dir_path, file_name)
    # bag = rosbag.Bag(bag).reindex()
    outbag = rosbag.Bag(new_bag, 'w')
    for topic, msg, t in rosbag.Bag(bag).read_messages():
        # This also replaces tf timestamps under the assumption
        # that all transforms in the message share the same timestamp
        print(t)
        print(type(t))
        now = datetime.fromtimestamp(t.secs + t.nsecs*1e-9)
        print(now)
        outbag.write(topic, msg, msg.header.stamp)
    outbag.close()
    outbag.reindex()

    return new_bag

def split_rosbag(inputbag,num_msgs=800):
    i = 1
    num_init = num_msgs
    ori = inputbag.split('.')[0]
    rosbag_dir =os.path.dirname(inputbag)
    print(argdict["bag_in"])
    inputdata = rosbag.Bag(inputbag)
    outputbag = ori + '_' + str(i) + '.bag'
    outdata = rosbag.Bag(outputbag, 'w')

    for topic, msg, t in inputdata.read_messages():

        if num_msgs:
            outdata.write(topic, msg, t)
            num_msgs -= 1
        else:
            i += 1
            # it is necessary ,or will have error : "ERROR bag unindexed: qiu2022-09-16-13-04-46_1.bag.  Run rosbag reindex."
            outdata.close()
            outdata.reindex()
            outputbag = ori + '_' + str(i) + '.bag'
            num_msgs = num_init
            outdata = rosbag.Bag(outputbag, 'w')
    outdata.close()
    outdata.reindex()
    return rosbag_dir


if __name__ == "__main__":
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description="Label point clouds with bounding boxes.")
    parser.add_argument("bag_in", type=str, help="Bag to load")
    parser.add_argument("calibration", type=str, help="cali_json to load")
    parser.add_argument("param_file", type=str, help="Yaml file matching topics and tf frames to channel names")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)

    # replace ros timstamp of timestamp in msg header
    # inputbag = rosbagStamp(argdict["bag_in"])
    # os.remove(argdict['bag_in'])
    # #
    # # split big rosbag into multiple small rosbag
    # argdict["bag_in"] = split_rosbag(inputbag, num_msgs=800)
    # os.remove(inputbag)


    # bags_list = all_path(argdict["bag_in"])
    # print(bags_list)
    # for entry in bags_list:
    argdict["bag_in"] = "/home/qiu/bagdir2/2023-01-07-13-23-38_3.bag"
    # argdict["bag_in"] = inputbag
    # argdict["bag_in"] = entry
    write_scene(argdict)