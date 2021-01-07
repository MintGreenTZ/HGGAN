from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from manopth.manolayer import ManoLayer

# Modified form manopth/demo.py and meshreg/datastes/ho3dv2.py
# Run show_hand_mano and ensure mano_root is correct.Default:assets/mano(in root)
# Use output_path='' to modify the output.
# Exsample:
# show_hand_mano(np.zeros(48),np.zeros(3),np.zeros(10))
def show_hand_mano(handpose, handtrans, handbeta, object=None, output_path='test.png', mano_root='assets/mano'):
    Info=trans_hand_verts(handpose, handtrans, handbeta)
    mano_layer=ManoLayer(mano_root=mano_root)
    display_hand(Info, mano_faces=mano_layer.th_faces, alpha=0.5, show=False, output_path=output_path, object=object)

def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True, output_path='test.png', object=None):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    if not(object is None):
        ax.scatter(object[:, 0], object[:, 1], object[:, 2], alpha=0.05)
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()
    else:
        plt.savefig(output_path)


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def trans_hand_verts(handpose, handtrans, handshape):
    Layer=ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root="assets/mano",
            center_idx=None,
            flat_hand_mean=True,
        )
    handverts, handjoints = Layer(
        torch.Tensor(handpose).unsqueeze(0), torch.Tensor(handshape).unsqueeze(0)
    )
    t_handverts = handverts[0].numpy() / 1000 + handtrans
    cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    trans_handverts = cam_extr[:3, :3].dot(t_handverts.transpose()).transpose()
    handverts[0]=torch.from_numpy(trans_handverts)
    return {'verts': handverts, 'joints': handjoints}