import argparse
import json
import os
import pickle
import pcl
import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
import open3d as o3d
def parse_args():
    parser = argparse.ArgumentParser(description='change nusc to xtreme format for labeling')

    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')

    args = parser.parse_args()
    return args

def save_pcd(pc: np.ndarray, file, binary=True):
    pc = pc.astype(np.float32)
    num_points = len(pc)

    with open(file, 'wb' if binary else 'w') as f:
        # heads
        headers = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            'FIELDS x y z i',
            'SIZE 4 4 4 4',
            'TYPE F F F F',
            'COUNT 1 1 1 1',
            f'WIDTH {num_points}',
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            f'POINTS {num_points}',
            f'DATA {"binary" if binary else "ascii"}'
        ]
        header = '\n'.join(headers) + '\n'
        if binary:
            header = bytes(header, 'ascii')
        f.write(header)

        # points
        if binary:
            f.write(pc.tobytes())
        else:
            for num in range(num_points):
                x, y, z, i = pc[num]
                f.write(f"{x:.3f} {y:.3f} {z:.3f}  {i:.3f}\n")
def main():
    args = parse_args()
    #mkdir
    root_dir = "/home/qiu/singleframe"
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    for i in range(6):
        img_dir = "/image"+str(i)
        if not os.path.isdir(root_dir + img_dir):
            os.mkdir(root_dir + img_dir)
    if not os.path.isdir(root_dir+"/camera_config"):
        os.mkdir(root_dir+"/camera_config")
    if not os.path.isdir(root_dir+"/point_cloud"):
        os.mkdir(root_dir+"/point_cloud")

    views = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    # load pkl，get infos
    info_path = \
            args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))


    # remember index acording to  data[infos]
    for i ,info in enumerate(dataset['infos']):
        print(f"process {i} info")
        lidar_points = np.fromfile(info['lidar_path'], dtype=np.float32)
        # aa = o3d.io.read_point_cloud(info['lidar_path'],format='auto')
        lidar_reshape = lidar_points.reshape(-1,5)
        lidar_reshape = lidar_reshape[:,:4]
        # aa = o3d.geometry.PointCloud()
        # aa.points = o3d.utility.Vector3dVector(lidar_reshape)

        if i < 10:
            # pcl.save(lidar_points,root_dir+"/point_cloud"+"/00000"+str(i)+".pcd")
            # o3d.io.write_point_cloud(root_dir+"/point_cloud"+"/00000"+str(i)+".pcd", aa)
            # lidar_points.tofile(root_dir+"/point_cloud"+"/00000"+str(i)+".pcd")
            save_pcd(lidar_reshape, root_dir+"/point_cloud"+"/00000"+str(i)+".pcd", binary=False)
        else:
            save_pcd(lidar_reshape, root_dir+"/point_cloud"+"/0000"+str(i)+".pcd", binary=False)
        for imgi,view in enumerate(views):
            img = cv2.imread(info['cams'][view]['data_path'])
            if i < 10:
                cv2.imwrite(root_dir+"/image"+str(imgi)+"/00000"+str(i)+".jpg",img)
            else:
                cv2.imwrite(root_dir + "/image" + str(imgi) + "/0000" + str(i) + ".jpg", img)
        cali_json = []
        for view in views:
            cam_cali_json = {}
            camrera_info = info['cams'][view]
            camera2lidar = np.eye(4, dtype=np.float32)
            camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
            camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
            lidar2camera = np.linalg.inv(camera2lidar)
            lidar2camera = lidar2camera.reshape(16).tolist()
            cam_cali_json["camera_external"] = lidar2camera
            cam_cali_json["width"] = 1600
            cam_cali_json["height"] = 900
            cam_cali_json["camera_internal"] = {}
            cam_cali_json["camera_internal"]["fx"] = camrera_info["cam_intrinsic"][0][0]
            cam_cali_json["camera_internal"]["fy"] = camrera_info["cam_intrinsic"][1][1]
            cam_cali_json["camera_internal"]["cx"] = camrera_info["cam_intrinsic"][0][2]
            cam_cali_json["camera_internal"]["cy"] = camrera_info["cam_intrinsic"][1][2]
            cali_json.append(cam_cali_json)
        if i < 10:
            json_file_path = root_dir + "/camera_config" + "/00000" + str(i) + ".json"
        else:
            json_file_path = root_dir + "/camera_config" + "/0000" + str(i) + ".json"

        with open(json_file_path, 'w', encoding='utf-8') as file_obj:
            json.dump(cali_json, file_obj, ensure_ascii=False)
        
if __name__== '__main__':
    main()