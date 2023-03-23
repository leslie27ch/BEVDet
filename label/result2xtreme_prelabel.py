import argparse
import json
import os
import pickle
import pcl
import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
from nanoid import generate

def generate_trackid():
    result = generate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",16)
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='change nusc to xtreme format for labeling')
    parser.add_argument(
        'res',
        type=str,
        help='result.json')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')


    args = parser.parse_args()
    return args

def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def main():
    args = parse_args()
    # load predicted results
    res = json.load(open(args.res, 'r'))
    # mkdir /home/qiu/singleframe/result
    res_dir = "/home/qiu/singleframe/result"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    all_res = res["results"].items()
    # delete score < 0.5
    for i, (sam_token, keyframe) in enumerate(all_res):
        keylist = keyframe.copy()
        for box in keylist:
            if box["detection_score"] < 0.5:
                keyframe.remove(box)

    info_path = \
        args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))

    for cnt, infos in enumerate(dataset['infos']):

        keyframe = res['results'][infos['token']]

        l2g = get_lidar2global(infos)
        g2l = np.linalg.inv(l2g).T

        # label_json = []
        dict_json={}
        obj_list = []
        # label_json.append(dict_json)

        # dict_json["sourceType"] = "EXTERNAL_GROUND_TRUTH"
        dict_json["objects"] = obj_list
        for obj_i,obj in enumerate(keyframe):
            #from bottom center to gravity center
            obj["translation"][2] += obj["size"][2]/2
            center = np.array(obj["translation"]).reshape(3)
            center = np.concatenate([center,np.ones([1])],axis=0)

            center_lidar = center @ g2l
            center_lidar = center_lidar.tolist()


            ego_rotaion = Quaternion(infos['ego2global_rotation']).inverse * obj["rotation"]
            lidar_rotaion = Quaternion(infos['lidar2ego_rotation']).inverse * ego_rotaion
            yaw_lidar = lidar_rotaion.yaw_pitch_roll[0]



            obj_dict = {}
            obj_dict["id"] = ""
            obj_dict["type"] = "3D_BOX"
            obj_dict["trackId"] = generate_trackid()
            obj_dict["trackName"] = str(obj_i+1)
            contour_dict = {}
            obj_dict["contour"] = contour_dict

            center_dict ={}
            contour_dict["center3D"] = center_dict


            center_dict["x"] = center_lidar[0]
            center_dict["y"] = center_lidar[1]
            center_dict["z"] = center_lidar[2]

            contour_dict["pointN"] = 0
            contour_dict["points"] = []

            rotate_dict = {}
            contour_dict["rotation3D"] = rotate_dict
            rotate_dict["x"] = 0
            rotate_dict["y"] = 0
            rotate_dict["z"] = yaw_lidar

            size3D_dict = {}

            #from wlh to lwh ,store lwh
            contour_dict["size3D"] = size3D_dict
            size3D_dict["x"] = obj["size"][1]
            size3D_dict["y"] = obj["size"][0]
            size3D_dict["z"] = obj["size"][2]

            obj_dict["modelConfidence"] = None
            obj_dict["modelClass"] = ""
            obj_dict["className"] = obj["detection_name"]

            obj_list.append(obj_dict)
        if cnt < 10:
            json_file_path = res_dir + "/00000" + str(cnt) + ".json"
        else:
            json_file_path = res_dir + "/0000" + str(cnt) + ".json"
        with open(json_file_path, 'w', encoding='utf-8') as file_obj:
            json.dump(dict_json, file_obj, ensure_ascii=False)


# def test_trackid():
#     a = generate_trackid()
#     print(a)
if __name__ == '__main__':
    # a = generate_trackid()
    # print(a)
    main()
