from datasets import (
    ho3dv2
)
import open3d
import pickle
import numpy as np
from tqdm import tqdm
DIS = 0.001
dataset = ho3dv2.HO3DV2(
    split="trainval",
    like_v1=False)
list_i=[]
for i in tqdm(range(len(dataset.idxs))):
    hand_o3=open3d.utility.Vector3dVector(dataset.get_hand_verts3d(i))
    obj_o3=open3d.utility.Vector3dVector(dataset.get_obj_verts_trans(i))
    hand_cloud=open3d.geometry.PointCloud(hand_o3)
    obj_cloud=open3d.geometry.PointCloud(obj_o3)
    list_of_dis=np.asarray(hand_cloud.compute_point_cloud_distance(obj_cloud))
    if np.min(list_of_dis) < DIS :
        list_i.append(i)
    # print(str(i) + "/" + str(len(dataset.idxs)))
with open ("filtered.pickle", 'wb') as f2:
    pickle.dump(list_i, f2)
print("original len = " + str(len(dataset.idxs)))
print("total len = " + str(len(list_i)))
