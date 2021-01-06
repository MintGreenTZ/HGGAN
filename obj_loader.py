import numpy as np


class Obj_Loader:
	def __init__(self, aligned = True):
		self.obj_model_file_names = ['obj_000.npy', 'obj_001.npy', 'obj_003.npy', 'obj_004.npy',
								'obj_005.npy', 'obj_007.npy', 'obj_008.npy', 'obj_009.npy']
		self.obj_model_names = [
			'003_cracker_box',
			'004_sugar_box',
			'006_mustard_bottle',
			'010_potted_meat_can',
			'011_banana',
			'025_mug',
			'035_power_drill',
			'037_scissors'
			]
		self.labels = [
			['MC2', 'MC4', 'MC5', 'MC6'],
			['SS1', 'SS3', 'ShSu12', 'ShSu13', 'ShSu14', 'Shsu', 'SS', 'SiS'],
			['SM2', 'SM4', 'SM5', 'SM'],
			['GPMF10', 'GPMF11', 'GPMF14', 'GPMF'],
			['BB10', 'BB13', 'BB14', 'SiBF10', 'SiBF12', 'SiBF13', 'SiBF14', 'SiBF', 'BB'],
			['SMu40', 'SMu41', 'SMu42', 'Smu'],
			['MDF10', 'MDF11', 'MDF12', 'MDF13', 'MDF', 'ND2'],
			['GSF10', 'GSF11', 'GSF12', 'GSF']
			]
		self.obj_models = []
		for i in range(len(self.obj_model_file_names)):
			self.obj_models.append(np.load("./data/obj_models/" + self.obj_model_file_names[i]))
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