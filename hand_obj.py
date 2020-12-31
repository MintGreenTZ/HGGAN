from datasets.ho3dv2 import HO3DV2
import open3d
import pickle
import numpy as np

HO3D = HO3DV2(split="trainval", like_v1=False)

print(HO3D.fulls)
print(type(HO3D.fulls))
print(len(HO3D.fulls))
print(HO3D.fulls[0])
print(type(HO3D.idxs))
# print(HO3D.idxs)
print(len(HO3D.idxs))
print(len(HO3D.idxs[0]))

rot, trans = HO3D.get_obj_pose(0)
# objTrans: A 3x1 vector representing object translation
# objRot: A 3x1 vector representing object rotation in Rodrigues representation
print(type(rot))
print(type(trans))
print(rot.shape) # (3, 3)
print(trans.shape) # (3,)

handpose, handtrans, handshape = HO3D.get_hand_info(0)
print(type(handpose))
print(type(handtrans))
print(type(handshape))
print(handpose.shape) # (48,)
print(handtrans.shape) # (3,)
print(handshape.shape) # (10,)
# handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints
# 	including the root joint in axis-angle representation. The ordering of
# 	the joints follow the MANO model convention (see joint_order.png) and 
# 	can be directly fed to MANO model.
# handTrans: A 3x1 vector representing the hand translation
# handBeta: A 10x1 vector representing the MANO hand shape parameters