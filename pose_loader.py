from datasets.ho3dv2 import HO3DV2
import open3d
import pickle
import numpy as np
import random
from obj_loader import Obj_Loader

class Pose_Loader:
	def __init__(self):
		self.HO3D = HO3DV2(split="trainval", like_v1=False)
		with open ("filtered.pickle", 'rb') as f:
			self.filtered_list = pickle.load(f)
		self.length = len(self.filtered_list)
		self.obj_loader = Obj_Loader(aligned = True)
		self.condition = np.stack([self.get_condition(i) for i in range(self.length)])
		self.true_pose = np.stack([self.get_true_pose(i) for i in range(self.length)])
		self.data = zip(self.condition, self.true_pose)
		self.pointer = 0

	def __len__(self):
		return self.length

	def get_obj(self, idx):
		# obj_idx \in [0, 8) obj_cloud (262146, 3)
		kind, kind_idx = self.HO3D.idxs[self.filtered_list[idx]]
		obj_idx, obj_cloud = self.obj_loader.get(kind)

	def get_obj_pose(self, idx):
		return self.HO3D.get_obj_pose(self.filtered_list[idx])

	def get_hand_info(self, idx):
		return self.HO3D.get_hand_info(self.filtered_list[idx])

	def get_condition(self, idx):
		# (3, 3) -> (9,) 	(1,) + (9,) + (3,) = (13,)
		obj_idx, obj_cloud = self.get_obj(idx)
		rot, trans = self.get_obj_pose(idx)
		return np.concatenate((np.array(obj_idx), rot.reshape((-1)), trans), axis = 0)

	def get_true_pose(self, idx):
		# (48,) + (3,) + (10,) = (61,)
		handpose, handtrans, handshape = HO3D.get_hand_info(0)
		return np.concatenate((handpose, handtrans, handshape), axis = 0)

	def get_condition_shape(self):
		return self.get_condition(0).shape

	def get_true_pose_shape(self):
		return self.get_true_pose(0).shape

	def next_batch(self, batch_size = 64):
		if self.pointer + batch_size > self.length:
			random.shuffle(self.data)
			self.pointer = 0
		self.pointer += batch_size
		return self.data[self.pointer - batch_size: self.pointer]
