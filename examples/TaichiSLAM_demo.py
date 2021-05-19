from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
from taichi_slam.utils.ros_pcl_transfer import *
import numpy as np
import math
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Point32, PoseStamped
import ros_numpy


cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = False

def taichioctomap_pcl_callback(octomap, cur_trans, msg):
    if cur_trans is None:
        return
    if octomap.TEXTURE_ENABLED:
        xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(msg)
    else:
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        rgb_array = np.array([], dtype=int)

    _R, _T = transform_msg_to_numpy(cur_trans)
    if project_in_taichi:
        for i in range(3):
            octomap.input_T[None][i] = _T[i]
            for j in range(3):
                octomap.input_R[None][i, j] = _R[i, j]
        octomap.recast_pcl_to_map(xyz_array, rgb_array, len(xyz_array), False)
    else:
        pts = []
        for pt in xyz_array:
            p = _R.dot(pt) + _T
            pts.append([p[0], p[1], p[2]])
        pts = np.array(pts)
        octomap.recast_pcl_to_map(pts, rgb_array, len(pts), True)

    global level
    if disp_in_rviz:
        pub_to_ros(pub, octomap.x.to_numpy(), octomap.color.to_numpy())
    level, pos_ = handle_render(octomap, scene, gui, pars, level)

def pub_to_ros(pub, pos_, colors_):
    octomap.get_voxel_to_particles(level)
    pcl = PointCloud()
    if octomap.TEXTURE_ENABLED:
        pts = np.concatenate((pos_, colors_.astype(float)/255.0), axis=1)
        pub.publish(point_cloud(pts, '/world', has_rgb=True))
    else:
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
    RES_X = 1024
    RES_Y = 768
    gui = ti.GUI('TaichiOctomap', (RES_X, RES_Y))
    level = 2
    scene = tina.Scene(RES_X, RES_Y, bgcolor=0xDDDDDD)
    pars = tina.SimpleParticles()
    material = tina.Lamp()
    scene.add_object(pars, material)
    octomap = Octomap()
    scene.init_control(gui, radius=octomap.map_scale_xy*0.7, theta=-1.57,center=(0, 0, 0), is_ortho=True)
    
    if disp_in_rviz:
        rospy.init_node("TaichiOctomap", disable_signals=False)
        pub = rospy.Publisher('/pcl', PointCloud2, queue_size=10)

    iteration_over_bag('/home/xuhao/data/voxblox/cow_and_lady_dataset.bag', 
        lambda _1, _2: taichioctomap_pcl_callback(octomap, _1, _2))
