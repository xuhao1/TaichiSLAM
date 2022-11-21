#!/usr/bin/env python3
from cmath import nan
import os, sys
sys.path.insert(0,os.path.dirname(__file__) + "/../")

from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
from taichi_slam.utils.ros_pcl_transfer import *
from taichi_slam.utils.communication import *
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import ros_numpy
import time
import message_filters
import cv2
from transformations import *
from swarm_msgs.msg import DroneTraj, VIOFrame

cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = False
count = 0

class TaichiSLAMNode:
    mesher: MarchingCubeMesher
    mapping: BaseMap
    def __init__(self):
        self.init_params()
        self.init_topology_generator() #Must multithread beforce init taichi
        if self.cuda:
            ti.init(arch=ti.cuda, dynamic_index=True, offline_cache=True, packed=True, debug=False, device_memory_GB=1.5)
        else:
            ti.init(arch=ti.cpu, dynamic_index=True, offline_cache=True, packed=True, debug=False)
        self.disp_level = 0
        self.count = 0
        self.cur_frame = None
        
        if self.enable_rendering:
            RES_X = rospy.get_param('~disp/res_x', 1920)
            RES_Y = rospy.get_param('~disp/res_y', 1080)
            self.render = TaichiSLAMRender(RES_X, RES_Y)
            self.render.enable_mesher = self.enable_mesher
            self.render.pcl_radius = rospy.get_param('~voxel_size', 0.05)/2

        self.pub_occ = rospy.Publisher('/dense_mapping', PointCloud2, queue_size=10)
        # self.pub_tsdf_surface = rospy.Publisher('/pub_tsdf_surface', PointCloud2, queue_size=10)

        self.updated = False
        self.initial_networking()
        self.initial_mapping()
        self.init_subscribers()
        self.updated_pcl = False
        self.post_submap_fusion_count = 0

    def init_params(self):
        self.cuda = rospy.get_param('~use_cuda', True)
        self.texture_compressed = rospy.get_param('~texture_compressed', False)
        self.enable_mesher = rospy.get_param('~enable_mesher', True)
        self.enable_rendering = rospy.get_param('~enable_rendering', True)
        self.output_map = rospy.get_param('~output_map', False)
        self.enable_submap = rospy.get_param('~enable_submap', False)
        self.enable_multi = rospy.get_param('~enable_multi', True)
        self.drone_id = rospy.get_param('~drone_id', 1)
        self.keyframe_step = rospy.get_param('~keyframe_step', 10) #Steps to generate submap
            
        fx_dep = rospy.get_param('Kdepth/fx', 384.2377014160156)
        fy_dep = rospy.get_param('Kdepth/fy', 384.2377014160156)
        cx_dep = rospy.get_param('Kdepth/cx', 323.4873046875)
        cy_dep = rospy.get_param('Kdepth/cy', 235.0628204345703)

        fx_color = rospy.get_param('Kcolor/fx', 384.2377014160156)
        fy_color = rospy.get_param('Kcolor/fy', 384.2377014160156)
        cx_color = rospy.get_param('Kcolor/cx', 323.4873046875)
        cy_color = rospy.get_param('Kcolor/cy', 235.0628204345703)
        #For L515
        self.Kdep = np.array([fx_dep, 0.0, cx_dep, 0.0, fy_dep, cy_dep, 0.0, 0.0, 1.0])
        self.Kcolor = np.array([fx_color, 0.0, cx_color, 0.0, fy_color, cy_color, 0.0, 0.0, 1.0])
        self.mapping_type = rospy.get_param('~mapping_type', 'tsdf')
        self.texture_enabled = rospy.get_param('~texture_enabled', True)
        self.max_mesh = rospy.get_param('~disp/max_mesh', 1000000)

        self.skeleton_graph_gen = rospy.get_param('~enable_skeleton_graph_gen', False)
        self.skeleton_graph_gen_opts = {}
        self.skeleton_graph_gen_opts['max_raycast_dist'] = rospy.get_param('~skeleton_graph_gen/max_raycast_dist', 2.5)
        self.skeleton_graph_gen_opts['coll_det_num'] = rospy.get_param('~skeleton_graph_gen/coll_det_num', 64)
        self.skeleton_graph_gen_opts['frontier_combine_angle_threshold'] = rospy.get_param('~skeleton_graph_gen/frontier_combine_angle_threshold', 20)
    
    def send_submap_handle(self, buf):
        self.comm.publishBuffer(buf, CHANNEL_SUBMAP)
    
    def traj_send_handle(self, traj):
        self.comm.publishBuffer(traj, CHANNEL_TRAJ)

    def initial_networking(self):
        if not self.enable_multi:
            return
        self.comm = SLAMComm(self.drone_id)
        self.comm.on_submap = self.on_remote_submap
        self.comm.on_traj = self.on_remote_traj
    
    def handle_comm(self):
        if not self.enable_multi:
            return
        self.comm.handle()
    
    def on_remote_submap(self, buf):
        self.mapping.input_remote_submap(buf)
    
    def on_remote_traj(self, buf):
        self.mapping.input_remote_traj(buf)

    def init_subscribers(self):
        self.depth_sub = message_filters.Subscriber('~depth', Image, queue_size=10)
        self.pointcloud_sub = message_filters.Subscriber('~pointcloud', PointCloud2, queue_size=10)

        if self.enable_submap:
            self.frame_sub = message_filters.Subscriber('~frame_local', VIOFrame)
            self.traj_sub = rospy.Subscriber("~traj", DroneTraj, self.traj_callback, queue_size=10, tcp_nodelay=True)
            if self.texture_enabled:
                if self.texture_compressed:
                    self.image_sub = message_filters.Subscriber('~image', CompressedImage, queue_size=10)
                else:
                    self.image_sub = message_filters.Subscriber('~image', Image)
                self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.image_sub, self.frame_sub], 10, slop=0.03)
                self.ts.registerCallback(self.process_depth_image_frame)
            else:
                self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.frame_sub], 10, slop=0.03)
                self.ts.registerCallback(self.process_depth_frame)
            self.ts_pcl = message_filters.ApproximateTimeSynchronizer([self.pointcloud_sub, self.frame_sub], 10, slop=0.03)
            self.ts_pcl.registerCallback(self.process_pcl_frame)
        else:
            self.pose_sub = message_filters.Subscriber('~pose', PoseStamped)
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

    def get_general_mapping_opts(self):
        max_disp_particles = rospy.get_param('~disp/max_disp_particles', 8000000)
        occupy_thres = rospy.get_param('~occupy_thres', 0)
        map_size_xy = rospy.get_param('~map_size_xy', 100)
        map_size_z = rospy.get_param('~map_size_z', 10)
        self.voxel_size = voxel_size = rospy.get_param('~voxel_size', 0.05)
        max_ray_length = rospy.get_param('~max_ray_length', 5.1)
        min_ray_length = rospy.get_param('~min_ray_length', 0.3)
        disp_ceiling = rospy.get_param('~disp_ceiling', 1.8)
        disp_floor = rospy.get_param('~disp_floor', -0.3)
        octo_opts = {
            'texture_enabled': self.texture_enabled, 
            'max_disp_particles': max_disp_particles, 
            'map_scale':[map_size_xy, map_size_z],
            'voxel_size':voxel_size,
            'max_ray_length':max_ray_length,
            'min_ray_length':min_ray_length,
            'disp_ceiling':disp_ceiling,
            'disp_floor':disp_floor,
            "texture_enabled": self.texture_enabled
        }
        return octo_opts

    def get_octo_opts(self):
        opts = self.get_general_mapping_opts()
        opts['K'] = rospy.get_param('K', 2)
        opts['min_occupy_thres'] = rospy.get_param('min_occupy_thres', 2)
        return opts
    
    def get_sdf_opts(self):
        opts = self.get_general_mapping_opts()
        opts.update({
            'num_voxel_per_blk_axis': rospy.get_param('~num_voxel_per_blk_axis', 16),  #How many voxels per block per axis
        })
        return opts

    def get_submap_opts(self):
        if self.mapping_type == "octo":
            opts = self.get_octo_opts()
        else:
            opts = self.get_sdf_opts()
        opts.update({
            'max_disp_particles': rospy.get_param('~submap_max_disp_particles', 100000),
        })
        return opts

    def initial_mapping(self):
        if self.enable_submap:
            print(f"Initializing submap with {self.mapping_type}...")
            if self.mapping_type == "octo":
                gopts = self.get_octo_opts()
                subopts = self.get_submap_opts()
                self.mapping = SubmapMapping(Octomap, global_opts=gopts, sub_opts=subopts, keyframe_step=self.keyframe_step)
                self.mapping.post_local_to_global_callback = self.post_submapfusion_callback
            else:
                gopts = self.get_sdf_opts()
                subopts = self.get_submap_opts()
                self.mapping = SubmapMapping(DenseTSDF, global_opts=gopts, sub_opts=subopts, keyframe_step=self.keyframe_step)
                self.mapping.post_local_to_global_callback = self.post_submapfusion_callback
                if self.enable_mesher:
                    self.mesher = MarchingCubeMesher(self.mapping.submap_collection, self.max_mesh, tsdf_surface_thres=self.voxel_size*5)
            self.mapping.map_send_handle = self.send_submap_handle
            self.mapping.traj_send_handle = self.traj_send_handle
        else:
            if self.mapping_type == "octo":
                opts = self.get_octo_opts()
                self.mapping = Octomap(**opts)
            elif self.mapping_type == "esdf" or self.mapping_type == "tsdf":
                opts = self.get_sdf_opts()
                self.mapping = DenseTSDF(**opts)
                if self.enable_mesher:
                    self.mesher = MarchingCubeMesher(self.mapping, self.max_mesh, tsdf_surface_thres=self.voxel_size*5)
        self.mapping.set_color_camera_intrinsic(self.Kcolor)
        self.mapping.set_dep_camera_intrinsic(self.Kdep)

            
    def init_topology_generator(self):
        if not self.skeleton_graph_gen:
            self.topo = None
        print("Initializing skeleton graph generator thread...")
        from multiprocessing import Process, Manager
        from topo_gen_thread import TopoGenThread
        self.share_map_man = Manager()
        self.shared_map_d = self.share_map_man.dict()
        self.shared_map_d["exit"] = False
        self.shared_map_d["update"] = False
        self.shared_map_d["topo_graph_viz"] = None
        params = {
            "sdf_params": self.get_sdf_opts(),
            "octo_params": self.get_octo_opts(),
            "skeleton_graph_gen_opts": self.skeleton_graph_gen_opts,
            "use_cuda": False #self.cuda
        }
        self.topo = Process(target=TopoGenThread, args=[params, self.shared_map_d])
        self.topo.start()

    def end_topo_thread(self):
        print("Ending topology thread...")
        self.shared_map_d["exit"] = True
        if self.topo:
            self.topo.terminate()
            self.topo.join()
            self.topo = None

    def process_depth_frame(self, depth_msg, frame):
        self.taichimapping_depth_callback(frame, depth_msg)

    def process_pcl_frame(self, cloud_msg, frame):
        self.cloud_msg = cloud_msg
        self.cur_frame = frame
        self.updated = True
        self.updated_pcl = True
        print("Updated PCL+Frame")

    def process_depth_image_frame(self, depth_msg, image, frame):
        if type(image) == CompressedImage:
            np_arr = np.frombuffer(image.data, np.uint8)
            rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            np_arr = np.frombuffer(image.data, np.uint8)
            np_arr = np_arr.reshape((image.height, image.width, -1))
            rgb_image = np_arr
        self.taichimapping_depth_callback(frame, depth_msg, rgb_image)

    def process_depth_pose(self, depth_msg, pose):
        #TODO: frame from pose
        pass
        
    def process_depth_image_pose(self, depth_msg, image, pose):
        #TODO: frame from pose
        if type(image) == CompressedImage:
            np_arr = np.frombuffer(image.data, np.uint8)
            rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            #self.texture_image = cv2.cvtColor(self.texture_image, cv2.COLOR_BGR2RGB)
        else:
            np_arr = np.frombuffer(image.data, np.uint8)
            rgb_image = np_arr.reshape((image.height, image.width, -1))
        # self.taichimapping_depth_callback(frame, depth_msg, rgb_image)

    def process_pcl_pose(self, msg, pose):
        if self.texture_enabled:
            xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(msg)
        else:
            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            rgb_array = np.array([], dtype=int)
        self.taichimapping_pcl_callback(pose, xyz_array, rgb_array)
    
    def rendering(self):
        start_time = time.time()
        mapping = self.mapping
        if self.mapping_type == "tsdf" and self.enable_rendering:
            if self.render.enable_slice_z:
                mapping.cvt_TSDF_to_voxels_slice(self.render.slice_z)
            else:
                mapping.cvt_TSDF_surface_to_voxels()
            self.render.set_particles(mapping.export_TSDF_xyz, mapping.export_color, mapping.num_TSDF_particles[None])
        self.render.rendering()
        return (time.time() - start_time)*1000
    
    def taichimapping_depth_callback(self, frame, depth_msg, texture=np.array([], dtype=int)):
        self.depth_msg = depth_msg
        self.cur_frame = frame
        self.texture = texture
        self.updated = True
            

    def taichimapping_pcl_callback(self, pose, xyz_array, rgb_array=None):
        pass
    
    def output(self, R, T):
        mapping = self.mapping
        t_mesh = nan
        t_export = nan
        t_pubros = nan
        if self.mapping_type == "octo":
            mapping.cvt_occupy_to_voxels(self.disp_level)
            par_count = mapping.num_export_particles[None]
            if self.output_map:
                self.pub_to_ros(mapping.export_x.to_numpy()[:par_count], 
                    mapping.export_color.to_numpy()[:par_count], mapping.enable_texture)
            if self.enable_rendering:
                self.render.set_particles(mapping.export_x, mapping.export_color)
        else:
            if self.enable_rendering and self.render.enable_mesher:
                mesher = self.mesher
                start_time = time.time()
                mesher.generate_mesh(1)
                t_mesh = (time.time() - start_time)*1000
                if self.enable_rendering:
                    self.render.set_mesh(mesher.mesh_vertices, mesher.mesh_colors, mesher.mesh_normals, mesher.num_vetices[None])
            else:
                if self.output_map:
                    start_time = time.time()
                    mapping.cvt_TSDF_surface_to_voxels()
                    t_export = (time.time() - start_time)*1000
                    par_count = mapping.num_TSDF_particles[None]
                    start_time = time.time()
                    self.pub_to_ros(mapping.export_TSDF_xyz.to_numpy()[:par_count], 
                            mapping.export_color.to_numpy()[:par_count], mapping.enable_texture)
                    t_pubros = (time.time() - start_time)*1000
        if self.enable_rendering and self.render.lock_pos_drone:
            self.render.camera_lookat = T
        return t_mesh, t_export, t_pubros
    
    def recast(self):
        frame = self.cur_frame
        mapping = self.mapping
        start_time = time.time()
        if self.updated_pcl:
            self.updated_pcl = False
            if self.texture_enabled:
                xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(self.cloud_msg)
            else:
                xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.cloud_msg)
                rgb_array = np.array([], dtype=int)
            t_pcl2npy = (time.time() - start_time)*1000
            if self.enable_submap:
                pose = pose_msg_to_numpy(frame.odom.pose.pose)
                frame_id = frame.frame_id
            else:
                pose = self.cur_pose
                frame_id = self.cur_frame_id
            ext = np.eye(3), np.zeros(3)
            mapping.recast_pcl_to_map_by_frame(frame_id, frame.is_keyframe, pose, ext, xyz_array, rgb_array)
        else:
            texture = self.texture
            w = self.depth_msg.width
            h = self.depth_msg.height
            depthmap = np.frombuffer(self.depth_msg.data, dtype=np.uint16)
            depthmap = np.reshape(depthmap, (h, w))
            t_pcl2npy = (time.time() - start_time)*1000
            if self.enable_submap:
                pose = pose_msg_to_numpy(frame.odom.pose.pose)
                frame_id = frame.frame_id
                ext = pose_msg_to_numpy(frame.extrinsics[0])
            else:
                pose = self.cur_pose
                frame_id = self.cur_frame_id
                ext = self.cur_ext
            mapping.recast_depth_to_map_by_frame(frame_id, frame.is_keyframe, pose, ext, depthmap, texture)
        return pose, t_pcl2npy, (time.time() - start_time)*1000

    def process_taichi(self):
        if not self.updated:
            return
        self.updated = False
        pose, t_pcl2npy, t_recast = self.recast()
        self.drone_pose_odom = pose
        if self.enable_rendering:
            self.render.set_drone_pose(0, pose[0], pose[1])
        t_mesh, t_export, t_pubros = self.output(pose[0], pose[1])
        self.count += 1
        print(f"[TaichiSLAM] Time: pcl2npy {t_pcl2npy:.1f}ms t_recast {t_recast:.1f}ms ms t_export {t_export:.1f}ms t_mesh {t_mesh:.1f}ms t_pubros {t_pubros:.1f}ms")
    
    def traj_callback(self, traj):
        if traj.drone_id != self.drone_id:
            return
        frame_poses = {}
        positions = np.zeros((len(traj.poses), 3))
        for i in range(len(traj.frame_ids)):
            R, T = pose_msg_to_numpy(traj.poses[i])
            frame_poses[traj.frame_ids[i]] = (R, T)
            positions[i] = T
        self.mapping.set_frame_poses(frame_poses)
        if self.enable_rendering:
            self.render.set_drone_trajectory(0, positions)

    def pub_to_ros(self, pos_, colors_, enable_texture):
        if enable_texture:
            pts = np.concatenate((pos_, colors_.astype(float)), axis=1)
            self.pub_occ.publish(point_cloud(pts, 'world', has_rgb=enable_texture))
        else:
            self.pub_occ.publish(point_cloud(pos_, 'world', has_rgb=enable_texture))
    
    def post_submapfusion_callback(self, global_map: BaseMap):
        self.post_submap_fusion_count += 1
        if self.topo:
            #Share the global map with shared memory
            obj = global_map.export_submap()
            self.shared_map_d["map_data"]= obj
            self.shared_map_d['update'] = True
            # print("Invoking topo skeleton generation")
            if self.shared_map_d["topo_graph_viz"] is not None:
                lines = self.shared_map_d["topo_graph_viz"]["lines"]
                self.render.set_skeleton_graph_edges(lines)
                
def slam_main():
    rospy.init_node( 'taichislam_node' )
    taichislamnode = TaichiSLAMNode()
    print("TaichiSLAMNode initialized")
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        try:
            taichislamnode.process_taichi()
            taichislamnode.handle_comm()
            if taichislamnode.enable_rendering:
                taichislamnode.rendering()
            rate.sleep()
        except KeyboardInterrupt:
            break
    taichislamnode.end_topo_thread()

if __name__ == '__main__':
    slam_main()
    

