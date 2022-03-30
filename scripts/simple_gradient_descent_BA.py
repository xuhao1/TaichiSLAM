import taichi as ti
import os, sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.taichi_transformations import *
from transformations import *
import matplotlib.pyplot as plt

ti.init()

Np = 10 # Poses
Nl = 100 # landmarks
Nzmax = Np*Nl #observation
max_iter = 100
rate = 3e-3

sigma_z = 3/385 # sigma of observation
sigma_L = 0.3 # sigma of landmark positions
sigma_T = 0.1 #sigma of camera position

q_poses = ti.Vector.field(n=4, dtype=ti.f32, shape=Np, needs_grad=True) #Quaternion of poses
T_poses = ti.Vector.field(n=3, dtype=ti.f32, shape=Np, needs_grad=True) #Translation of poses
L_p = ti.Vector.field(n=3, dtype=ti.f32, shape=Nl, needs_grad=True) #Landmark positions

Z_l = ti.Vector.field(n=2, dtype=ti.f32, shape=Nzmax, needs_grad=True) #Landmark oberservations, in unit vector
Z_pind = ti.field(dtype=ti.i32, shape=Nzmax)
Z_lind = ti.field(dtype=ti.i32, shape=Nzmax)

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
Nz = ti.field(dtype=ti.i32, shape=(), needs_grad=False)


Lgt = []
Pgt = []
Qgt = []

@ti.func
def project_to_image(T):
    return ti.Vector([T[0]/T[2], T[1]/T[2]])

@ti.func 
def reprojection_err(z_l, l_p, q, T):
    l_ = l_p - T
    l_b = QuaternionRotate(QuaternionInverse(q), l_)
    z = project_to_image(l_b)
    e =  z - z_l
    # print("reprojection_err", z, z_l, e[0]*e[0] + e[1] * e[1])
    return e[0]*e[0] + e[1] * e[1]

@ti.kernel
def func():
    for i in range(Nz[None]):
        ind_p = Z_pind[i]
        ind_l = Z_lind[i]
        err = reprojection_err(Z_l[i], L_p[ind_l], q_poses[ind_p], T_poses[ind_p])
        loss[None] += err/Nz[None]

def generate_poses():
    global Pgt, Qgt
    for i in range(Np):
        pgt = [i/10. , 0., 0.]
        p = np.random.normal(pgt, sigma_T, size=(3))

        T_poses[i] = ti.Vector(p)
        q_poses[i] = ti.Vector([0., 0.1, 0., 1.])
        q_poses[i] = q_poses[i]/q_poses[i].norm()

        Pgt.append(pgt)
        Qgt.append([0., 0, 0, 1.]) #Attenti
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
            if np.fabs(x) < 1 and np.fabs(y) < 1:
                Z_l[c] = ti.Vector([x, y])
                Z_lind[c] = i
                Z_pind[c] = j
                c += 1
    Nz[None] = c
    
@ti.func
def gradient_descent():
    for i in range(Np):
        T_poses[i] -= rate*T_poses.grad[i]
        grad_tangent = q_poses.grad[i].transpose()@PlusQuaternionJacobian(q_poses[i])
        # print("q", q_poses[i], q_poses.grad[i], grad_tangent)
        q_poses[i] = QuaternionRetraction(q_poses[i], -rate*grad_tangent.transpose()) #Need retraction
    for i in range(Nl):
        L_p[i] = L_p[i] - rate*L_p.grad[i]

@ti.kernel
def iteration():
    loss[None] = 0
    loss.grad[None] = 1
    func()
    func.grad()
    gradient_descent()

np.random.seed(0)
generate_poses()
generate_landmarks()
generate_observations()

loss.grad[None] = 1
func()
func.grad()
print("Initial loss", loss[None])
# print(q_poses.grad)
# print("Gradient of T_poses:\n", T_poses.grad)
# print("Gradient of L_p:\n", L_p.grad)

loss_log = []
for i in range(max_iter):
    iteration()
    func()
    loss_log.append(loss[None])
    # print("Iter", i, "loss", loss[None])

plt.figure("loss")
plt.title("loss")
ticks = range(len(loss_log))
plt.plot(ticks, loss_log)
plt.legend()
plt.grid()
plt.show()