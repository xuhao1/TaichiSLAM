from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
from taichi_slam.utils.ros_pcl_transfer import *
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointCloud
import ros_numpy
import tina
import time

cur_trans = None
pub = None
project_in_taichi = True
disp_in_rviz = False

def rendering(mapping):
    global level
    level, t_v2p = mapping.handle_render(scene, gui, pars1, level, pars_sdf=pars2, substeps = 1)
    return t_v2p

def taichimapping_pcl_callback(mapping, cur_trans, msg, enable_rendering):
    if cur_trans is None:
        return

    start_time = time.time()
    if mapping.TEXTURE_ENABLED:
        xyz_array, rgb_array = pointcloud2_to_xyz_rgb_array(msg)
    else:
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        rgb_array = np.array([], dtype=int)

    _R, _T = transform_msg_to_numpy(cur_trans)

    for i in range(3):
        mapping.input_T[None][i] = _T[i]
        for j in range(3):
            mapping.input_R[None][i, j] = _R[i, j]

    t_pcl2npy = (time.time() - start_time)*1000
    start_time = time.time()

    mapping.recast_pcl_to_map(xyz_array, rgb_array, len(xyz_array))
    t_recast = (time.time() - start_time)*1000

    start_time = time.time()
    if disp_in_rviz:
        pub_to_ros(pub, mapping.export_x.to_numpy(), mapping.export_color.to_numpy(), mapping.TEXTURE_ENABLED)
    t_pubros = (time.time() - start_time)*1000

    start_time = time.time()
    t_v2p = 0
    if enable_rendering:
        t_v2p = rendering(mapping)
    t_render = (time.time() - start_time)*1000

    print(f"Time: pcl2npy {t_pcl2npy:.1f}ms t_recast {t_recast:.1f}ms t_v2p {t_v2p:.1f}ms t_pubros {t_pubros:.1f}ms t_render {t_render:.1f}ms")

def pub_to_ros(pub, pos_, colors_, TEXTURE_ENABLED):
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
    import argparse
    parser = argparse.ArgumentParser(description='Taichi slam fast demo')
    parser.add_argument("-r","--resolution", nargs=2, type=int, help="display resolution", default=[1024, 768])
    parser.add_argument("-m","--method", type=str, help="dense mapping method: octo/esdf", default="octo")
    parser.add_argument("-c","--cuda", help="enable cuda acceleration if applicable", action='store_true')
    parser.add_argument("-t","--texture-enabled", help="showing the point cloud's texture", action='store_true')
    parser.add_argument("--rviz", help="output to rviz", action='store_true')
    parser.add_argument("-p", "--max-disp-particles", help="max output voxels", type=int,default=1000000)
    parser.add_argument("-b","--bagpath", help="path of bag", type=str,default='')
    parser.add_argument("-o","--occupy-thres", help="thresold for occupy", type=int,default=2)
    parser.add_argument("-s","--map-size", help="size of map xy,z in meter", nargs=2, type=float, default=[100, 10])
    parser.add_argument("--blk", help="block size of esdf, if blk==1; then dense", type=int, default=32)
    parser.add_argument("-v","--voxel-size", help="size of voxel", type=float, default=0.05)
    parser.add_argument("-K", help="division each axis of octomap, when K>2, octomap will be K**3-map", type=int, default=2)
    parser.add_argument("-f", "--rendering-final", help="only rendering the final state", action='store_true')
    parser.add_argument("--record", help="record to C code", action='store_true')

    args = parser.parse_args()

    RES_X = args.resolution[0]
    RES_Y = args.resolution[1]
    disp_in_rviz = args.rviz
    
    print()
    print(f"Res [{RES_X}x{RES_Y}] GPU {args.cuda} RVIZ {disp_in_rviz} size of map {args.map_size} grid {args.voxel_size} ")

    if args.record:
        ti.core.start_recording('./export/TaichiSLAM.yml')
        ti.init(arch=ti.cc)
    else:
        if args.cuda:
            ti.init(arch=ti.cuda)
        else:
            ti.init(arch=ti.cpu)


    gui = ti.GUI('TaichiSLAM', (RES_X, RES_Y))
    level = 1
    scene = tina.Scene(RES_X, RES_Y, bgcolor=(0.1, 0.1, 0.1))
    pars1 = tina.SimpleParticles(maxpars=args.max_disp_particles)
    pars2 = tina.SimpleParticles(maxpars=args.max_disp_particles)
    material1 = tina.Lamp()
    material2 = tina.Lamp()
    scene.add_object(pars1, material1)
    scene.add_object(pars2, material2)
    if args.method == "octo":
        mapping = Octomap(texture_enabled=args.texture_enabled, 
            max_disp_particles=args.max_disp_particles, 
            min_occupy_thres = args.occupy_thres,
            map_scale=args.map_size,
            voxel_size=args.voxel_size,
            K=args.K)
    elif args.method == "esdf":
        mapping = DenseESDF(texture_enabled=args.texture_enabled, 
            max_disp_particles=args.max_disp_particles, 
            min_occupy_thres = args.occupy_thres,
            map_scale=args.map_size,
            voxel_size=args.voxel_size,
            block_size=args.blk)

    scene.init_control(gui, radius=6, theta=-math.pi/4,center=(0, 0, 0), is_ortho=True)
    
    if disp_in_rviz:
        rospy.init_node("Taichimapping", disable_signals=False)
        pub = rospy.Publisher('/pcl', PointCloud2, queue_size=10)

    if args.bagpath == "":
        print("No data input, using random generate maps")
        mapping.random_init_octo(1000)
        while gui.running:
            rendering(mapping)    
    else:
        iteration_over_bag(args.bagpath, 
            lambda _1, _2: taichimapping_pcl_callback(mapping, _1, _2, not args.rendering_final))

    while gui.running:
        rendering(mapping)    


