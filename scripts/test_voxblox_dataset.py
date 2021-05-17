from taichi_octomap import *
import numpy as np
import math
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointCloud
import ros_numpy

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


cur_trans = None

def pcl_callback(msg):
    if cur_trans is None:
        return
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)[::5,:]
    R, T = transform_msg_to_numpy(cur_trans)
    for i in range(3):
        input_T[None][i] = T[i]
        for j in range(3):
            input_R[None][i, j] = R[i, j]
    recast_pcl_to_map(xyz_array, len(xyz_array))
    global level
    level = handle_render(scene, gui, pars, level)


def pose_call_back(msg):
    global cur_trans
    cur_trans = msg

if __name__ == '__main__':
    RES = 1024
    rospy.init_node("TaichiOctomap", disable_signals=False)
    gui = ti.GUI('TaichiOctomap', (RES, RES))
    level = R//2
    scene = tina.Scene(RES)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale, theta=-1.3, center=(0, 0, 0))
    Broot.deactivate_all()
    sub2 = rospy.Subscriber("/kinect/vrpn_client/estimated_transform", TransformStamped, pose_call_back)
    sub1 = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pcl_callback)

    r = rospy.Rate(10) # 10hz 
    while not rospy.is_shutdown():
        try:
            r.sleep()
        except KeyboardInterrupt:
            break
