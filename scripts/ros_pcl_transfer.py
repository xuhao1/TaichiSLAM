import rosbag
import matplotlib.pyplot as plt
import ros_numpy
import numpy as np
import math
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Point32, PoseStamped
import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

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

    Rdb = np.array([
        [0.971048, -0.120915, 0.206023, 0.00114049],
        [0.15701, 0.973037, -0.168959, 0.0450936],
        [-0.180038, 0.196415, 0.96385, 0.0430765],
        [0, 0, 0, 1]
    ])
    R[0:3,3] = T
    R = np.matmul(R, Rdb)
    T = R[0:3, 3]
    return R[0:3,0:3], T

def point_cloud(points, parent_frame, has_rgb=False):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()
    
    channels = 'xyzrgba'
    steps = 7
    if not has_rgb:
        channels = "xyz"
        steps = 3

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(channels)]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * steps),
        row_step=(itemsize * steps * points.shape[0]),
        data=data
    )

def pcl_callback(cur_trans, msg):
    pose = PoseStamped()

    pose.header.stamp = cur_trans.header.stamp
    pose.pose.position = cur_trans.transform.translation
    pose.pose.orientation = cur_trans.transform.rotation
    pose.header.frame_id = "world"

    pts = []
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)[::5,:]
    R, T = transform_msg_to_numpy(cur_trans)
    for pt in xyz_array:
        p = R.dot(pt) + T
        pts.append([p[0], p[1], p[2]])
    pts = np.array(pts)
    pub.publish(point_cloud(pts, '/world'))
    if pub_pose is not None:
        pub_pose.publish(pose)

def iteration_over_bag(path, callback):
    bag = rosbag.Bag(path)
    cur_trans = None
    count_depth = 0
    for topic, msg, t in bag.read_messages(topics=['/camera/depth_registered/points', '/kinect/vrpn_client/estimated_transform']):
        try:
            if topic == '/camera/depth_registered/points':
                callback(cur_trans, msg)
            else:
                cur_trans = msg
        except KeyboardInterrupt:
            exit(0)
            break

if __name__ == "__main__":
    rospy.init_node("TaichiOctomap", disable_signals=False)
    pub = rospy.Publisher('/pcl', PointCloud2, queue_size=10)
    pub_pose = rospy.Publisher('/pose', PoseStamped, queue_size=10)
    iteration_over_bag('/home/xuhao/data/voxblox/data.bag', pcl_callback)
