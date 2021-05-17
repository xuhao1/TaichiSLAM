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
disp_in_rviz = False

def render_map_to_particles(pars, pos_, colors, num_particles_, level):
    pos = pos_[0:num_particles_,:]
    if not TEXTURE_ENABLED:
        max_z = np.max(pos[:,2])
        min_z = np.min(pos[:,2])
        colors = cm.jet((pos[:,2] - min_z)/(max_z-min_z))
    pars.set_particles(pos)
    radius = np.ones(num_particles_)*(K**(level-1))*grid_scale
    pars.set_particle_radii(radius)
    pars.set_particle_colors(colors)

def handle_render(scene, gui, pars, level):
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == "-":
            level += 1
            if level == R:
                level = R - 1
        elif e.key == "=":
            level -= 1
            if level < 0:
                level = 0
    get_voxel_to_particles(level)
    pos_ = x.to_numpy()
    color_ = color.to_numpy()
    render_map_to_particles(pars, pos_, color_, num_export_particles[None], level)

    for i in range(3):
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.text(content=f'Level {level:.2f} num_particles {num_export_particles[None]} grid_scale {(K**(level))*grid_scale} incress =; decress -',
                pos=(0, 0.8),
                font_size=20,
                color=0x080808)

        gui.show()
    return level, pos_
    
def taichioctomap_pcl_callback(cur_trans, msg):
    if cur_trans is None:
        return
    if TEXTURE_ENABLED:
        xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(msg)
    else:
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        rgb_array = np.array([], dtype=int)

    _R, _T = transform_msg_to_numpy(cur_trans)
    if project_in_taichi:
        for i in range(3):
            input_T[None][i] = _T[i]
            for j in range(3):
                input_R[None][i, j] = _R[i, j]
        recast_pcl_to_map(xyz_array, rgb_array, len(xyz_array))
    else:
        pts = []
        for pt in xyz_array:
            p = _R.dot(pt) + _T
            pts.append([p[0], p[1], p[2]])
        pts = np.array(pts)
        recast_pcl_to_map_no_project(pts, len(pts))

    global level
    if disp_in_rviz:
        pub_to_ros(pub, x.to_numpy(), color.to_numpy())
    level, pos_ = handle_render(scene, gui, pars, level)

def pub_to_ros(pub, pos_, colors_):
    get_voxel_to_particles(level)
    pcl = PointCloud()
    if TEXTURE_ENABLED:
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
    RES_X = 1920
    RES_Y = 1080
    gui = ti.GUI('TaichiOctomap', (RES_X, RES_Y))
    level = 1
    scene = tina.Scene(RES_X, RES_Y, bgcolor=0xDDDDDD)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale, theta=-1.57,center=(0, 0, 0))
    Broot.deactivate_all()
    rospy.init_node("TaichiOctomap", disable_signals=False)
    pub = rospy.Publisher('/pcl', PointCloud2, queue_size=10)

    iteration_over_bag('/home/xuhao/data/voxblox/cow_and_lady_dataset.bag', taichioctomap_pcl_callback)
