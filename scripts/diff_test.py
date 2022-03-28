import taichi as ti
import os, sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.taichi_transformations import *
ti.init()

Np = 10 # Poses
Nl = 10 # landmarks
Nz = Np*Nl #observation
max_iter = 100
rate = 1e-5

sigma_z = 0.1 # sigma of observation
sigma_L = 0.3 # sigma of landmark positions

q_poses = ti.Vector.field(n=4, dtype=ti.f32, shape=Np, needs_grad=True) #Quaternion of poses
T_poses = ti.Vector.field(n=3, dtype=ti.f32, shape=Np, needs_grad=True) #Translation of poses
L_p = ti.Vector.field(n=3, dtype=ti.f32, shape=Nl, needs_grad=True) #Landmark positions

Z_l = ti.Vector.field(n=2, dtype=ti.f32, shape=Nz, needs_grad=True) #Landmark oberservations, in unit vector
Z_pind = ti.field(dtype=ti.i32, shape=Nz)
Z_lind = ti.field(dtype=ti.i32, shape=Nz)

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


Lgt = []
Pgt = []
Qgt = []
@ti.func
def project_to_image(T):
    return ti.Vector([T[0]/T[2], T[1]/T[2]])

@ti.func 
def reprojection_err(z_l, l_p, q, T):
    T_ = l_p - T
    T_b = quaternion_rotate(quaternion_inverse(q), T_)
    z = project_to_image(T_b)
    # print("reproject obs", z)
    e =  z - z_l
    return e[0]*e[0] + e[1] * e[1]

@ti.kernel
def func():
    for i in range(Nz):
        ind_p = Z_pind[i]
        ind_l = Z_lind[i]
        err = reprojection_err(Z_l[i], L_p[ind_l], q_poses[ind_p], T_poses[ind_p])
        # print("z", i, "pose", ind_p, "landmark", ind_l, "err", err)
        loss[None] += err

def generate_poses():
    global Pgt, Qgt
    for i in range(Np):
        q_poses[i] = ti.Vector([0., 0., 0., 1.])
        T_poses[i] = ti.Vector([i/10., 0., 0.])
        Pgt.append([i/10., 0., 0.])
        Qgt.append([0, 0, 0, 1]) #Attenti
    Pgt = np.array(Pgt)
    Qgt = np.array(Qgt)

def generate_landmarks():
    global Lgt
    Lgt = np.random.uniform([-5, -5, 0.5], [5, 5, 10], size=(Nl, 3))
    Lnoise = Lgt + np.random.normal(0, sigma_L, size=(Nl, 3))
    for i in range(Nl):
        L_p[i] = ti.Vector(Lnoise[i])

def generate_observations():
    c = 0
    for i in range(Nl):
        for j in range(Np):
            Pl = Lgt[i]
            p = Pgt[j]
            q = Qgt[j]
            q_inv = quaternion_inverse_(q)
            Pl = quaternion_rotate_(q_inv, Pl - p)
            x = np.random.normal(Pl[0] / Pl[2], sigma_z)
            y = np.random.normal(Pl[1] / Pl[2], sigma_z)
            Z_l[c] = ti.Vector([x, y])
            Z_lind[c] = i
            Z_pind[c] = j
            c += 1

@ti.kernel
def iteration():
    func()
    func.grad()
    for i in range(Np):
        T_poses[i] -= rate*T_poses.grad[i]
    for i in range(Nl):
        L_p[i] -= rate*L_p.grad[i]

generate_poses()
generate_landmarks()
generate_observations()

loss.grad[None] = 1

func()
func.grad()
print("Grad at begin")
print(q_poses.grad)
print(T_poses.grad)

for i in range(max_iter):
    iteration()
    func()
    print("Iter", i, "loss", loss[None])