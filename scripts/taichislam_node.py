#!/usr/bin/env python3
import os, sys
sys.path.insert(0,os.path.dirname(__file__) + "/../")

from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
from taichi_slam.utils.ros_pcl_transfer import *
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image, CameraInfo, CompressedImage
import ros_numpy
import time
import message_filters
import cv2
from transformations import *

cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = False
count = 0

class TaichiSLAMNode:
    def __init__(self):
        cuda = rospy.get_param('~use_cuda', True)
        self.mapping_type = rospy.get_param('~mapping_type', 'tsdf')
        self.texture_enabled = rospy.get_param('~texture_enabled', True)
        self.texture_compressed = rospy.get_param('~texture_compressed', False)
        self.enable_mesher = rospy.get_param('~enable_mesher', True)
        occupy_thres = rospy.get_param('~occupy_thres', 0)
        map_size_xy = rospy.get_param('~map_size_xy', 100)
        map_size_z = rospy.get_param('~map_size_z', 10)
        voxel_size = rospy.get_param('~voxel_size', 0.05)
        block_size = rospy.get_param('~block_size', 16)
        self.enable_rendering = rospy.get_param('~enable_rendering', True)
        self.output_map = rospy.get_param('~output_map', False)
        K = rospy.get_param('K', 2)
        max_disp_particles = rospy.get_param('~disp/max_disp_particles', 8000000)
        max_mesh = rospy.get_param('~disp/max_mesh', 1000000)
        max_ray_length = rospy.get_param('~max_ray_length', 5.1)
        min_ray_length = rospy.get_param('~min_ray_length', 0.3)
        
        if cuda:
            ti.init(arch=ti.cuda, device_memory_fraction=0.2, dynamic_index=True, offline_cache=True)
        else:
            ti.init(arch=ti.cpu, dynamic_index=True, offline_cache=True)

        self.disp_level = 0
        self.count = 0
        self.cur_pose = None ## Naive approach, need sync!!!
         
        if self.mapping_type == "octo":
            self.mapping = Octomap(texture_enabled=self.texture_enabled, 
                max_disp_particles=max_disp_particles, 
                min_occupy_thres = occupy_thres,
                map_scale=[map_size_xy, map_size_z],
                voxel_size=voxel_size,
                max_ray_length=max_ray_length,
                min_ray_length=min_ray_length,
                K=K)
        elif self.mapping_type == "esdf" or self.mapping_type == "tsdf":
            self.mapping = DenseESDF(texture_enabled=self.texture_enabled, 
                max_disp_particles=max_disp_particles, 
                min_occupy_thres = occupy_thres,
                map_scale=[map_size_xy, map_size_z],
                voxel_size=voxel_size,
                block_size=block_size,
                enable_esdf=self.mapping_type == "esdf",
                min_ray_length=min_ray_length,
                max_ray_length=max_ray_length)
            if self.enable_mesher:
                self.mesher = MarchingCubeMesher(self.mapping, max_mesh, tsdf_surface_thres=voxel_size*3)

        if self.enable_rendering:
            RES_X = rospy.get_param('~disp/res_x', 1920)
            RES_Y = rospy.get_param('~disp/res_y', 1080)
            self.render = TaichiSLAMRender(RES_X, RES_Y)
            self.render.enable_mesher = self.enable_mesher

        self.pub_occ = rospy.Publisher('/occ', PointCloud2, queue_size=10)
        self.pub_tsdf_surface = rospy.Publisher('/pub_tsdf_surface', PointCloud2, queue_size=10)
    
        self.pose_sub = message_filters.Subscriber('~pose', PoseStamped)
        self.depth_sub = message_filters.Subscriber('~depth', Image, queue_size=10)

        if self.texture_enabled:
            if self.texture_compressed:
                self.image_sub = message_filters.Subscriber('~image', CompressedImage, queue_size=10)
            else:
                self.image_sub = message_filters.Subscriber('~image', Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.image_sub, self.pose_sub], 10, slop=0.03)
            self.ts.registerCallback(self.process_depth_image_pose)

        else:
            self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.pose_sub], 10, slop=0.03)
            self.ts.registerCallback(self.process_depth_pose)

        fx_dep = rospy.get_param('Kdepth/fx', 384.2377014160156)
        fy_dep = rospy.get_param('Kdepth/fy', 384.2377014160156)
        cx_dep = rospy.get_param('Kdepth/cx', 323.4873046875)
        cy_dep = rospy.get_param('Kdepth/cy', 235.0628204345703)

        fx_color = rospy.get_param('Kcolor/fx', 384.2377014160156)
        fy_color = rospy.get_param('Kcolor/fy', 384.2377014160156)
        cx_color = rospy.get_param('Kcolor/cx', 323.4873046875)
        cy_color = rospy.get_param('Kcolor/cy', 235.0628204345703)

        #For L515
        self.K = np.array([fx_dep, 0.0, cx_dep, 0.0, fy_dep, cy_dep, 0.0, 0.0, 1.0])
        self.Kcolor = np.array([fx_color, 0.0, cx_color, 0.0, fy_color, cy_color, 0.0, 0.0, 1.0])

        # self.test_mesher()

    def test_mesher(self):
        self.mapping.init_sphere()
        self.mesher.generate_mesh(1)
        mesher = self.mesher
        self.render.set_particles(mesher.mesh_vertices, mesher.mesh_vertices)
        self.render.set_mesh(mesher.mesh_vertices, mesher.mesh_colors, mesher.mesh_normals, mesher.mesh_indices)
    
    def update_test_mesher(self):
        self.mapping.cvt_TSDF_to_voxels_slice(self.render.slice_z, 100)
        self.render.set_particles(self.mapping.export_TSDF_xyz, self.mapping.export_color)


    def process_depth_pose(self, depth_msg, pose):
        self.cur_pose = pose
        self.depth_msg = depth_msg
        self.rgb_array = np.array([], dtype=int)
        self.texture_image = np.array([], dtype=int)
        
    def process_depth_image_pose(self, depth_msg, image, pose):
        if type(image) == CompressedImage:
            np_arr = np.fromstring(image.data, np.uint8)
            self.texture_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            np_arr = np.fromstring(image.data, np.uint8)
            np_arr = np_arr.reshape((image.height, image.width, -1))
            self.texture_image = np_arr
        self.depth_msg = depth_msg
        self.rgb_array = np.array([], dtype=int)
        self.cur_pose = pose

    def process_pcl_pose(self, msg, pose):
        if self.texture_enabled:
            xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(msg)
        else:
            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            rgb_array = np.array([], dtype=int)
        # self.taichimapping_pcl_callback(pose, xyz_array, rgb_array)
        
    def update(self):
        if self.cur_pose is not None:
            self.taichimapping_depth_callback(self.cur_pose, self.depth_msg, self.rgb_array)
            self.cur_pose = None
        else:
            if self.mapping_type == "tsdf" and self.enable_rendering:
                if self.render.enable_slice_z:
                    self.mapping.cvt_TSDF_to_voxels_slice(self.render.slice_z)
                self.render.set_particles(self.mapping.export_TSDF_xyz, self.mapping.export_color)
                


    def taichimapping_depth_callback(self, pose, depth_msg, rgb_array=None):
        mapping = self.mapping

        start_time = time.time()

        _R, _T = pose_msg_to_numpy(pose.pose)
        
        mapping.set_pose(_R, _T)
        
        t_pcl2npy = (time.time() - start_time)*1000
        start_time = time.time()
        w = depth_msg.width
        h = depth_msg.height

        depthmap = np.frombuffer(depth_msg.data, dtype=np.uint16)
        depthmap = np.reshape(depthmap, (h, w))

        mapping.recast_depth_to_map(depthmap, self.texture_image, w, h, self.K, self.Kcolor)

        t_recast = (time.time() - start_time)*1000

        start_time = time.time()
        t_pubros = 0
        t_mesh = 0
        t_export = 0

        if self.mapping_type == "octo":
            mapping.cvt_occupy_to_voxels(self.disp_level)
            par_count = mapping.num_export_particles[None]
            if self.output_map:
                self.pub_to_ros(mapping.export_x.to_numpy()[:par_count], 
                    mapping.export_color.to_numpy()[:par_count], mapping.TEXTURE_ENABLED)
            if self.enable_rendering:
                self.render.set_particles(mapping.export_x, mapping.export_color)
        else:
            if self.render.enable_mesher:
                mesher = self.mesher
                start_time = time.time()
                mesher.generate_mesh(1)
                t_mesh = (time.time() - start_time)*1000
                if self.enable_rendering:
                    self.render.set_mesh(mesher.mesh_vertices, mesher.mesh_colors, mesher.mesh_normals)
            else:
                start_time = time.time()
                mapping.cvt_TSDF_surface_to_voxels()
                t_export = (time.time() - start_time)*1000

                par_count = mapping.num_export_TSDF_particles[None]

                if self.output_map:
                    start_time = time.time()
                    self.pub_to_ros(mapping.export_TSDF_xyz.to_numpy()[:par_count], 
                            mapping.export_color.to_numpy()[:par_count], mapping.TEXTURE_ENABLED)
                    t_pubros = (time.time() - start_time)*1000
                
                if self.enable_rendering:
                    self.render.set_particles(mapping.export_TSDF_xyz, mapping.export_color)

        if self.enable_rendering and self.render.lock_pos_drone:
            self.render.camera_lookat = _T


        start_time = time.time()
        t_v2p = 0
        t_render = (time.time() - start_time)*1000
        
        self.count += 1
        print(f"Time: pcl2npy {t_pcl2npy:.1f}ms t_recast {t_recast:.1f}ms ms t_v2p {t_v2p:.1f}ms t_export{t_export:.1f}ms t_mesh {t_mesh:.1f}ms t_pubros {t_pubros:.1f}ms t_render {t_render:.1f}ms")

    def taichimapping_pcl_callback(self, pose, xyz_array, rgb_array=None):
        mapping = self.mapping

        start_time = time.time()

        _R, _T = pose_msg_to_numpy(pose.pose)

        mapping.input_T = ti.Vector(_T)
        mapping.input_R = ti.Matrix(_R)

        t_pcl2npy = (time.time() - start_time)*1000
        start_time = time.time()
        mapping.recast_pcl_to_map(xyz_array, rgb_array, len(xyz_array))
        
        t_recast = (time.time() - start_time)*1000

        start_time = time.time()
        if self.mapping_type == "octo":
            mapping.cvt_occupy_to_voxels(self.disp_level)

        if self.output_map:
            self.pub_to_ros(mapping.export_x.to_numpy(), mapping.export_color.to_numpy(), mapping.TEXTURE_ENABLED)
        t_pubros = (time.time() - start_time)*1000

        start_time = time.time()
        t_v2p = 0
        t_render = (time.time() - start_time)*1000
        
        self.count += 1
        print(f"Time: pcl2npy {t_pcl2npy:.1f}ms t_recast {t_recast:.1f}ms ms t_v2p {t_v2p:.1f}ms t_pubros {t_pubros:.1f}ms t_render {t_render:.1f}ms")

    def pub_to_ros_surface(self, pos_, colors_, TEXTURE_ENABLED):
        if TEXTURE_ENABLED:
            pts = np.concatenate((pos_, colors_.astype(float)), axis=1)
            self.pub_occ.publish(point_cloud(pts, '/world', has_rgb=TEXTURE_ENABLED))
        else:
            self.pub_occ.publish(point_cloud(pos_, '/world', has_rgb=TEXTURE_ENABLED))

    def pub_to_ros(self, pos_, colors_, TEXTURE_ENABLED):
        if TEXTURE_ENABLED:
            pts = np.concatenate((pos_, colors_.astype(float)), axis=1)
            self.pub_occ.publish(point_cloud(pts, '/world', has_rgb=TEXTURE_ENABLED))
        else:
            self.pub_occ.publish(point_cloud(pos_, '/world', has_rgb=TEXTURE_ENABLED))


if __name__ == '__main__':
    
    rospy.init_node( 'taichislam_node' )

    taichislamnode = TaichiSLAMNode()
    print("TaichiSLAMNode initialized")
    
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        if taichislamnode.enable_rendering:
            taichislamnode.render.rendering()
        taichislamnode.update()
        # taichislamnode.update_test_mesher()
        rate.sleep()

