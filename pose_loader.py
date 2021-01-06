from datasets.ho3dv2 import HO3DV2
import open3d
import pickle
import numpy as np
import random
from obj_loader import Obj_Loader

class Pose_Loader:
	def __init__(self):
		self.HO3D = HO3DV2(split="trainval", like_v1=False)
		self.obj_loader = Obj_Loader(aligned = True)
		limited_kind_list = list(np.hstack(self.obj_loader.labels))
		self.filtered_list = []
		with open ("filtered.pickle", 'rb') as f:
			obj_dis_filtered_list = pickle.load(f)
		count = 0
		not_count = 0
		for i in range(len(obj_dis_filtered_list)):
			kind, kind_idx = self.HO3D.idxs[obj_dis_filtered_list[i]]
			if kind in limited_kind_list:
				count = count + 1
				self.filtered_list.append(obj_dis_filtered_list[i])
			else:
				not_count = not_count + 1
		print("valid number is " + str(count))
		print("rest number is " + str(not_count))

		self.length = len(self.filtered_list)
		self.condition = np.stack([self.get_condition(i) for i in range(self.length)])
		self.true_pose = np.stack([self.get_true_pose(i) for i in range(self.length)])
		self.data = zip(self.condition, self.true_pose)
		self.pointer = 0

	def __len__(self):
		return self.length

	def get_obj(self, idx):
		# obj_idx \in [0, 8) obj_cloud (262146, 3)
		kind, kind_idx = self.HO3D.idxs[self.filtered_list[idx]]
		# print("from get_obj get " + str(kind) + ", " + str(kind_idx))
		obj_idx, obj_cloud = self.obj_loader.get(kind)
		return obj_idx, obj_cloud

	def get_obj_pose(self, idx):
		return self.HO3D.get_obj_pose(self.filtered_list[idx])

	def get_hand_info(self, idx):
		return self.HO3D.get_hand_info(self.filtered_list[idx])

	def get_condition(self, idx):
		# (3, 3) -> (9,) 	(1,) + (9,) + (3,) = (13,)
		# print("get_condition " + str(idx))
		obj_idx, obj_cloud = self.get_obj(idx)
		rot, trans = self.get_obj_pose(idx)
		# print(np.array([obj_idx]))
		# print(rot.reshape((-1)))
		# print(trans)
		# print(np.concatenate((np.array([obj_idx]), rot.reshape((-1)), trans), axis = 0))
		return np.concatenate((np.array([obj_idx]), rot.reshape((-1)), trans), axis = 0)

	def get_true_pose(self, idx):
		# (48,) + (3,) + (10,) = (61,)
		handpose, handtrans, handshape = self.HO3D.get_hand_info(0)
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
