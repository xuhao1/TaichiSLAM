import taichi as ti
import numpy as np

#The order of taichi vector is follow x y z w
kW = 3
kX = 0
kY = 1
kZ = 2

@ti.func
def QuaternionMatrix(q):
    # p.115@Quaternion kinematics for the error-state KF
    qw = q[kW]
    qx = q[kX]
    qy = q[kY]
    qz = q[kZ]
    return ti.Matrix([[qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], 
        [2*(qx*qy + qw*qz), (qw*qw - qx*qx + qy*qy - qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy + qz*qz]])

@ti.func
def QuaternionInverse(q):
    return ti.Vector([-q.x, -q.y, -q.z, q.w])

@ti.func
def QuaternionRotate(q, v):
    R = QuaternionMatrix(q)
    v = R@v
    return v

@ti.func
def PlusQuaternionJacobian(q):
    qw = q[kW]
    qx = q[kX]
    qy = q[kY]
    qz = q[kZ]
    return ti.Matrix([
        [qw, qz, -qy],
        [-qz, qw, qx],
        [qy, -qx, qw],
        [-qx, -qy, -qz]
    ])

@ti.func
def QuaternionMultiply(q0, q1):
    w0, x0, y0, z0 = q0[kW], q0[kX], q0[kY], q0[kZ]
    w1, x1, y1, z1 = q1[kW], q1[kX], q1[kY], q1[kZ]
    return ti.Matrix([
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ])

@ti.func
def QuaternionRetraction(q, delta):
    norm_delta = delta.norm()
    qret = q
    if norm_delta != 0:
        sin_delta_by_delta = ti.sin(norm_delta) / norm_delta
        dq = ti.Vector([sin_delta_by_delta * delta[0], 
                        sin_delta_by_delta * delta[1], 
                        sin_delta_by_delta * delta[2], 
                        ti.cos(norm_delta)])
        qret = QuaternionMultiply(dq, q)
    return qret

def QuaternionInverse_(q):
    return [-q[kX], -q[kY], -q[kZ], q[kW]]

def QuaternionMatrix_(q):
    # p.115@Quaternion kinematics for the error-state KF
    qw = q[kW]
    qx = q[kX]
    qy = q[kY]
    qz = q[kZ]
    return np.array([[qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], 
        [2*(qx*qy + qw*qz), (qw*qw - qx*qx + qy*qy - qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy + qz*qz]])

def QuaternionRotate_(q, v):
    R = QuaternionMatrix_(q)
    v = R@v
    return v