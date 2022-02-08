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

cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = False
count = 0


class TaichiSLAMNode:
    def __init__(self):
        cuda = rospy.get_param('use_cuda', True)
        ti.init(device_memory_GB=4) 

        self.mapping_type = rospy.get_param('mapping_type', 'tsdf')
        self.texture_enabled = rospy.get_param('texture_enabled', True)
        self.texture_compressed = rospy.get_param('texture_compressed', True)
        occupy_thres = rospy.get_param('occupy_thres', 10)
        map_size_xy = rospy.get_param('map_size_xy', 100)
        map_size_z = rospy.get_param('map_size_z', 10)
        voxel_size = rospy.get_param('voxel_size', 0.05)
        block_size = rospy.get_param('block_size', 16)
        self.enable_rendering = rospy.get_param('enable_rendering', False)
        self.output_map = rospy.get_param('output_map', True)
        K = rospy.get_param('K', 2)
        max_disp_particles = rospy.get_param('disp/max_disp_particles', 1000000)
        max_ray_length = rospy.get_param('max_ray_length', 3.1)
        
        if cuda:
            ti.init(arch=ti.cuda)
        else:
            ti.init(arch=ti.cpu)

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
                K=K)
        elif self.mapping_type == "esdf" or self.mapping_type == "tsdf":
            self.mapping = DenseESDF(texture_enabled=self.texture_enabled, 
                max_disp_particles=max_disp_particles, 
                min_occupy_thres = occupy_thres,
                map_scale=[map_size_xy, map_size_z],
                voxel_size=voxel_size,
                block_size=block_size,
                max_ray_length=max_ray_length)

        if self.enable_rendering:
            self.init_gui()

        self.pub_occ = rospy.Publisher('/occ', PointCloud2, queue_size=10)
        self.pub_tsdf_surface = rospy.Publisher('/pub_tsdf_surface', PointCloud2, queue_size=10)
    
        # image_sub = message_filters.Subscriber('~depth', Image)
        # info_sub = message_filters.Subscriber('~pose', PoseStamped)

        self.depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image, queue_size=10)
        self.pose_sub = message_filters.Subscriber('/vins_estimator/camera_pose', PoseStamped, queue_size=10)

        if self.texture_enabled:
            if self.texture_compressed:
                self.image_sub = message_filters.Subscriber('/camera/infra1/image_rect_raw/compressed', CompressedImage, queue_size=10)
            else:
                self.image_sub = message_filters.Subscriber('/camera/infra1/image_rect_raw', Image)
            self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.image_sub, self.pose_sub], 10)
            self.ts.registerCallback(self.process_depth_image_pose)

        else:
            self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], 10)
            self.ts.registerCallback(self.process_depth_pose)

        self.K = np.array([384.2377014160156, 0.0, 323.4873046875, 0.0, 384.2377014160156, 235.0628204345703, 0.0, 0.0, 1.0])

    def init_gui(self):
        RES_X = rospy.get_param('disp/res_x', 1024)
        RES_Y = rospy.get_param('disp/res_y', 768)
        self.max_disp_particles = rospy.get_param('disp/max_disp_particles', 1000000)
        self.pcl_radius = rospy.get_param('disp/pcl_radius', 0.02)
        
        self.window = window = ti.ui.Window('TaichiSLAM', (RES_X, RES_Y), vsync=True)

        self.canvas = window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = camera = ti.ui.make_camera()

        camera.position(-2, -2, 2)
        camera.lookat(0, 0, 0)
        camera.up(0., 0., 1.)
        camera.fov(55)

    def options(self):
        window = self.window

        window.GUI.begin("Options", 0.05, 0.45, 0.2, 0.4)
        self.pcl_radius = window.GUI.slider_float("particles radius ",
                                            self.pcl_radius, 0, 0.1)
        self.disp_level = math.floor(window.GUI.slider_float("display level ",
                                            self.disp_level, 0, 10))
        window.GUI.end()

    def rendering(self):
        if not self.enable_rendering:
            return
        scene = self.scene
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.camera.up(0., 0., 1.)

        scene.set_camera(self.camera)
        
        scene.ambient_light((0, 0, 0))

        print(self.mapping.export_x)
        scene.particles(self.mapping.export_x, per_vertex_color=self.mapping.export_color, radius=self.pcl_radius)
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
        self.canvas.scene(scene)
        self.options()
        self.window.show()

    def process_depth_pose(self, depth_msg, pose):
        self.cur_pose = pose
        self.depth_msg = depth_msg
        self.rgb_array = np.array([], dtype=int)
        self.texture_image = np.array([], dtype=int)
        
    def process_depth_image_pose(self, depth_msg, image, pose):
        if type(image) == CompressedImage:
            np_arr = np.fromstring(image.data, np.uint8)
            self.texture_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

        mapping.recast_depth_to_map(depthmap, self.texture_image, w, h, self.K)

        t_recast = (time.time() - start_time)*1000

        start_time = time.time()
        if self.mapping_type == "octo":
            mapping.cvt_occupy_to_voxels(self.disp_level)
            par_count = mapping.num_export_particles[None]
            if self.output_map:
                self.pub_to_ros(mapping.export_x.to_numpy()[:par_count], 
                    mapping.export_color.to_numpy()[:par_count], mapping.TEXTURE_ENABLED)
        else:
            mapping.cvt_TSDF_surface_to_voxels()
            par_count = mapping.num_export_TSDF_particles[None]
            if self.output_map:
                self.pub_to_ros(mapping.export_TSDF_xyz.to_numpy()[:par_count], 
                    mapping.export_color.to_numpy()[:par_count], mapping.TEXTURE_ENABLED)

        t_pubros = (time.time() - start_time)*1000

        start_time = time.time()
        t_v2p = 0
        t_render = (time.time() - start_time)*1000
        
        self.count += 1
        print(f"Time: pcl2npy {t_pcl2npy:.1f}ms t_recast {t_recast:.1f}ms ms t_v2p {t_v2p:.1f}ms t_pubros {t_pubros:.1f}ms t_render {t_render:.1f}ms")

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
    rospy.init_node( 'TaichiSLAMNode' )

    taichislamnode = TaichiSLAMNode()
    
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        taichislamnode.rendering()
        taichislamnode.update()
        rate.sleep()

