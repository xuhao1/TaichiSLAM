import taichi as ti
import numpy as np
@ti.func
def quaternion_matrix(q):
    # p.115@Quaternion kinematics for the error-state KF
    qw = q.w
    qx = q.x
    qy = q.y
    qz = q.z
    return ti.Matrix([[qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], 
        [2*(qx*qy + qw*qz), (qw*qw - qx*qx + qy*qy - qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy + qz*qz]])

@ti.func
def quaternion_inverse(q):
    return ti.Vector([q.x, -q.y, -q.z, -q.w])

@ti.func
def quaternion_rotate(q, v):
    R = quaternion_matrix(q)
    v = R@v
    return v

def quaternion_inverse_(q):
    return [q[0], -q[1], -q[2], -q[3]]

def quaternion_matrix_(q):
    # p.115@Quaternion kinematics for the error-state KF
    qw = q[3]
    qx = q[0]
    qy = q[1]
    qz = q[2]
    return np.array([[qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], 
        [2*(qx*qy + qw*qz), (qw*qw - qx*qx + qy*qy - qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy + qz*qz]])

def quaternion_rotate_(q, v):
    R = quaternion_matrix_(q)
    v = R@v
    return v
