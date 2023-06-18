import argparse
import os
import sys
from io_util import read_pcd
from tensorpack import DataFlow, dataflow
import numpy as np
import scipy.io as sio
import pdb
from open3d import *


class pcd_df(DataFlow):
    def __init__(self, ex_times, num_scans, NBV_dir, gt_dir, data_type):
        self.num_scans = num_scans
        self.ex_times = ex_times
        self.NBV_dir = NBV_dir
        self.gt_dir = gt_dir
        self.data_type = data_type

    def size(self):
        if self.data_type == 'valid':
            return 4000
        elif self.data_type == 'train':
            return 42490
        elif self.data_type == 'test':
            return 2040
        elif self.data_type == 'test_novel':
            return 4000

    def get_data(self):
        for mock in range(1):
            model_list = os.listdir(self.gt_dir)
            for model_id in model_list:
                gt_pc = None
                points_cloud_path = os.path.join(self.gt_dir, model_id, 'model.pcd')
                if os.path.exists(points_cloud_path):
                    points_cloud = open3d.io.read_point_cloud(points_cloud_path)
                    gt_pc = np.asarray(points_cloud.points)
                else:
                    print("[ERROR] points_cloud_path not exist: {}".format(points_cloud_path))
                    sys.exit(1)
                
                for ex_index in range(self.ex_times):
                    for scan_index in range(self.num_scans):
                        view_state = None
                        accumulate_pointcloud = None
                        target_value = None
                        
                        view_state_path = os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_viewstate.npy")
                        if os.path.exists(view_state_path):
                            view_state = np.load(view_state_path) # shape (33) , 33 is view number
                        else:
                            print("[ERROR] view_state_path not exist: {}".format(view_state_path))
                            sys.exit(1)
                        
                        accumulate_pointcloud_path = os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_acc_pc.npy")
                        if os.path.exists(accumulate_pointcloud_path):
                            accumulate_pointcloud = np.load(accumulate_pointcloud_path) # shape (point number, 3)
                        else:
                            print("[ERROR] accumulate_pointcloud_path not exist: {}".format(accumulate_pointcloud_path))
                            sys.exit(1)
                        
                        target_value_path = os.path.join(self.NBV_dir, str(model_id), str(ex_index), str(scan_index) + "_target_value.npy")
                        if os.path.exists(target_value_path):
                            target_value = np.load(target_value_path) # shape (33, 1), 33 is view number
                        else:
                            print("[ERROR] target_value_path not exist: {}".format(target_value_path))
                            sys.exit(1)
                        
                        yield model_id, accumulate_pointcloud, gt_pc, view_state, target_value

if __name__ == '__main__':
    data_type = 'train'
    gt_dir = "/home/wang/data/tf/train/"
    output_path = "/mnt/data1/wang/tf/" + data_type + ".lmdb"
    NBV_dir = "/home/wang/data/tf/train_output/"
    ex_times = 1
    num_scans = 10

    df = pcd_df(ex_times, num_scans, NBV_dir, gt_dir, data_type)
    if os.path.exists(output_path):
        os.system('rm %s' % output_path)

    dataflow.LMDBSerializer.save(df, output_path)
