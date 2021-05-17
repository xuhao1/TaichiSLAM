from taichi_octomap import *
import numpy as np
import math

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def transform_msg_to_numpy(cur_trans):
    T = np.array([
        cur_trans.transform.translation.x,
        cur_trans.transform.translation.y,
        cur_trans.transform.translation.z
    ])

    R = quaternion_matrix([
        cur_trans.transform.rotation.x,
        cur_trans.transform.rotation.y,
        cur_trans.transform.rotation.z,
        cur_trans.transform.rotation.w
    ])

    return R, T


if __name__ == '__main__':
    RES = 1024
    gui = ti.GUI('TaichiOctomap', (RES, RES))
    level = R//2
    scene = tina.Scene(RES)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale/2, theta=-1.0, center=(map_scale/2, map_scale/2, map_scale/2))
    Broot.deactivate_all()

    import rosbag
    import sensor_msgs.point_cloud2 as pc2

    #Level = 0 most detailed
    bag = rosbag.Bag('/home/xuhao/data/voxblox/cow_and_lady_dataset.bag')
    cur_trans = None
    count_depth = 0
    for topic, msg, t in bag.read_messages(topics=['/camera/depth_registered/points', '/kinect/vrpn_client/estimated_transform']):
        if topic == '/camera/depth_registered/points':
            num_input_points[None] = 0
            k = 0
            for p in pc2.read_points_list(msg, field_names = ("x", "y", "z"), skip_nans=True):
                if k % 5 == 0:
                    pcl_input[num_input_points[None]][0] = p.x
                    pcl_input[num_input_points[None]][1] = p.y
                    pcl_input[num_input_points[None]][2] = p.z
                    num_input_points[None] += 1
                k += 1
            R, T = transform_msg_to_numpy(cur_trans)
            for i in range(3):
                input_T[None][i] = T[i]
                for j in range(3):
                    input_R[None][i, j] = R[i, j]
            print(T)
            recast_pcl_to_map()
            count_depth += 1
        else:
            cur_trans = msg
        level = handle_render(scene, gui, pars, level)