import numpy as np


class Obj_Loader:
	def __init__(self, aligned = True):
		self.obj_model_names = ['obj_000.npy', 'obj_001.npy', 'obj_003.npy', 'obj_004.npy',
								'obj_005.npy', 'obj_007.npy', 'obj_008.npy', 'obj_009.npy']
		self.obj_models = []
		self.labels = [['MC2'], ['Shsu', 'SS', 'SiS'], ['SM'], ['GPMF'],
						['SiBF', 'BB'], ['Smu'], ['MDF', 'ND2'], ['GSF']]
		for i in range(len(self.obj_model_names)):
			self.obj_models.append(np.load("./data/obj_models/" + self.obj_model_names[i]))
		if aligned:
			self.align()

	def __len__(self):
		return len(self.obj_models)

	def __getitem__(self, key):
		return self.obj_models[key]

	def get(self, key):
		for i, kinds in enumerate(self.labels):
			if key in kinds:
				return i, self.obj_models[i]
		raise Exception("No such data.")

	def align(self):
		length = len(self.obj_models)
		max_size = 0
		for i in range(length):
			max_size = max(max_size, self.obj_models[i].shape[0])
		for i in range(length):
			self.obj_models[i].resize((max_size, 3))