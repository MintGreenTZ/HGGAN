from datasets.ho3dv2 import HO3DV2
import open3d
import pickle
import numpy as np

class Pose_Loader:
	def __init__(self):
		self.HO3D = HO3DV2(split="trainval", like_v1=False)
		with open ("filtered.pickle", 'rb') as f:
    		self.filtered_list = pickle.load(f)

    def __len__(self):
    	return len(self.filtered_list)

    def get_obj_pose(self, idx):
    	return self.HO3D.get_obj_pose(self.filtered_list[idx])

    def get_hand_info(self, idx):
    	return self.HO3D.get_hand_info(self.filtered_list[idx])