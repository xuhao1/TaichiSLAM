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

class BetterRender:
    def __init__(self, RES_X, RES_Y):
        self.window = window = ti.ui.Window('TaichiSLAM', (RES_X, RES_Y), vsync=True)
        self.pcl_radius = 0.01

        self.canvas = window.get_canvas()
        self.canvas.set_background_color((207/255.0, 243/255.0, 250/255.0))
        self.scene = ti.ui.Scene()
        self.camera = camera = ti.ui.make_camera()
        camera.fov(55)
        
        self.par = None
        self.par_color = None

        self.camera_yaw = 0
        self.camera_pitch = -0.5
        self.camera_distance = 3
        self.camera_lookat = np.array([0., 0., 0.])
        self.camera_pitch_rate = 3.0
        self.camera_yaw_rate = 3.0
        self.camera_move_rate = 3.0
        self.scale_rate = 5
        self.lock_pos_drone = False

        self.set_camera_pose()

        self.mouse_last = None

        self.window.show()
    
    def set_camera_pose(self):
        pos = np.array([-self.camera_distance, 0., 0])
        pos = euler_matrix(0, -self.camera_pitch, -self.camera_yaw)[0:3,0:3]@pos + self.camera_lookat

        self.camera.position(pos[0], pos[1], pos[2])
        self.camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        self.camera.up(0., 0., 1.)
        self.scene.set_camera(self.camera)

    def options(self):
        window = self.window

        window.GUI.begin("Options", 0.05, 0.45, 0.2, 0.4)
        self.pcl_radius = window.GUI.slider_float("particles radius ",
                                            self.pcl_radius, 0.005, 0.03)
        self.lock_pos_drone = window.GUI.checkbox("Look Drone", self.lock_pos_drone)
        # self.disp_level = math.floor(window.GUI.slider_float("display level ",
        #                                     self.disp_level, 0, 10))
        window.GUI.end()
    
    def set_particles(self, par, color):
        self.par = par
        self.par_color = color

    def handle_events(self):
        win = self.window
        x, y = win.get_cursor_pos()

        if self.mouse_last is None:
            self.mouse_last = win.get_cursor_pos()
        x_s = self.mouse_last[0]
        y_s = self.mouse_last[1]

        if win.is_pressed(ti.ui.LMB):
            self.camera_pitch += self.camera_pitch_rate*(y-y_s)
            self.camera_yaw += self.camera_yaw_rate*(x-x_s)

        if win.is_pressed(ti.ui.MMB):
            R = euler_matrix(0, -self.camera_pitch, -self.camera_yaw)[0:3,0:3]
            move = self.camera_move_rate*self.camera_distance*np.array([0, x-x_s, -(y-y_s)])
            self.camera_lookat += R@move

        if win.is_pressed(ti.ui.RMB):
            move = self.scale_rate*(y-y_s)
            self.camera_distance += move
        
        self.camera_lookat[2] = 0.0 #Lock on XY
            # print(f"move {move} R@move {R@move}", self.camera_lookat)

        self.mouse_last = (x, y)

    def rendering(self):
        self.handle_events()
        self.set_camera_pose()
        
        scene = self.scene
        
        # self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.LMB)

        scene.ambient_light((1.0, 1.0, 1.0))

        if self.par is not None:
            scene.particles(self.par, per_vertex_color=self.par_color, radius=self.pcl_radius)
            
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        self.canvas.scene(scene)
        self.options()
        self.window.show()
        
class TaichiSLAMNode:
    def __init__(self):
        cuda = rospy.get_param('use_cuda', True)

        self.mapping_type = rospy.get_param('mapping_type', 'octo')
        self.texture_enabled = rospy.get_param('texture_enabled', True)
        self.texture_compressed = rospy.get_param('texture_compressed', True)
        occupy_thres = rospy.get_param('occupy_thres', 10)
        map_size_xy = rospy.get_param('map_size_xy', 100)
        map_size_z = rospy.get_param('map_size_z', 10)
        voxel_size = rospy.get_param('voxel_size', 0.03)
        block_size = rospy.get_param('block_size', 16)
        self.enable_rendering = rospy.get_param('enable_rendering', True)
        self.output_map = rospy.get_param('output_map', False)
        K = rospy.get_param('K', 2)
        max_disp_particles = rospy.get_param('disp/max_disp_particles', 8000000)
        max_ray_length = rospy.get_param('max_ray_length', 3.1)
        
        if cuda:
            ti.init(arch=ti.cuda, device_memory_fraction=0.6)
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
                enable_esdf=self.mapping_type == "esdf",
                max_ray_length=max_ray_length)

        if self.enable_rendering:
            RES_X = rospy.get_param('disp/res_x', 1920)
            RES_Y = rospy.get_param('disp/res_y', 1080)
            self.render = BetterRender(RES_X, RES_Y)

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
            if self.enable_rendering:
                self.render.set_particles(mapping.export_x, mapping.export_color)
        else:
            mapping.cvt_TSDF_surface_to_voxels()
            par_count = mapping.num_export_TSDF_particles[None]
            if self.output_map:
                self.pub_to_ros(mapping.export_TSDF_xyz.to_numpy()[:par_count], 
                    mapping.export_color.to_numpy()[:par_count], mapping.TEXTURE_ENABLED)
            if self.enable_rendering:
                self.render.set_particles(mapping.export_TSDF_xyz, mapping.export_color)

        if self.enable_rendering and self.render.lock_pos_drone:
            self.render.camera_lookat = _T

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
        if taichislamnode.enable_rendering:
            taichislamnode.render.rendering()
        taichislamnode.update()
        rate.sleep()

