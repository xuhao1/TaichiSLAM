from taichi_octomap import *
import numpy as np
import math
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Point32, PoseStamped
import ros_numpy

from ros_pcl_transfer import * 

cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = True

def taichioctomap_pcl_callback(cur_trans, msg):
    if cur_trans is None:
        return
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)[::5,:]
    _R, _T = transform_msg_to_numpy(cur_trans)
    if project_in_taichi:
        for i in range(3):
            input_T[None][i] = _T[i]
            for j in range(3):
                input_R[None][i, j] = _R[i, j]
        recast_pcl_to_map(xyz_array, len(xyz_array))
    else:
        pts = []
        for pt in xyz_array:
            p = _R.dot(pt) + _T
            pts.append([p[0], p[1], p[2]])
        pts = np.array(pts)
        recast_pcl_to_map_no_project(pts, len(pts))

    global level
    if disp_in_rviz:
        pub_to_ros(pub, x.to_numpy())
    else:
        level, pos_ = handle_render(scene, gui, pars, level)

def pub_to_ros(pub, pos_):
    get_voxel_to_particles(level)
    pcl = PointCloud()
    pub.publish(point_cloud(pos_, '/world'))


def pose_call_back(msg):
    global cur_trans
    cur_trans = msg

def ros_subscribe_pcl():
    sub2 = rospy.Subscriber("/kinect/vrpn_client/estimated_transform", TransformStamped, pose_call_back)
    sub1 = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pcl_callback)

    r = rospy.Rate(10) # 10hz 
    while not rospy.is_shutdown():
        try:
            r.sleep()
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    RES = 1024
    gui = ti.GUI('TaichiOctomap', (RES, RES))
    level = 1
    scene = tina.Scene(RES, bgcolor=0xDDDDDD)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale, theta=-1.57,center=(0, 0, 0))
    Broot.deactivate_all()
    rospy.init_node("TaichiOctomap", disable_signals=False)
    pub = rospy.Publisher('/pcl', PointCloud2, queue_size=10)

    iteration_over_bag('/home/xuhao/data/voxblox/data.bag', taichioctomap_pcl_callback)
