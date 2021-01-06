from datasets.ho3dv2 import HO3DV2
from pose_loader import Pose_Loader

loader = Pose_Loader()

dataset = HO3DV2(
	split="trainval",
    like_v1=False
    )
all_obj = {}
data_pack = {}
length = len(dataset.idxs)
for i in range(length):
	obj_idx, name = dataset.get_obj_name(i)
	all_obj[obj_idx] = name
	kind, kind_idx = dataset.idxs[i]
	if name in data_pack:
		data_pack[name][kind] = ""
	else:
		data_pack[name] = {}

print(all_obj)
print(data_pack)

# from obj_loader import Obj_Loader

# obj_loader = Obj_Loader(aligned = True)
# print(len(obj_loader))
# print(type(obj_loader[0]))
# for i in range(len(obj_loader)):
# 	print(obj_loader[i].shape)