import taichi as ti
import os, sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.taichi_transformations import *
from transformations import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *

ti.init()

Np = 16 # Poses
Nl = 100 # landmarks
Nzmax = Np*Nl #observation
max_iter = 1000
rate = 1.5e-1

sigma_z = 0/385 # sigma of observation
sigma_L = 0.1 # sigma of landmark positions
sigma_T = 0.02 #sigma of camera position
sigma_Q = 0.05 #sigma of camera quaternion

sqrt_inf_z = 1.

max_camera_pos = np.array([1, 1, 0])
min_camera_pos = np.array([-1, -1, 0])

max_landmark_pos = np.array([1.2, 1.2, 2])
min_landmark_pos = np.array([-1.2, -1.2, 0.5])

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
    e =  (z - z_l)*sqrt_inf_z
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
        pgt = np.array([i//sqrt(Np), i%sqrt(Np), 0.])
        pgt = pgt/sqrt(Np) * (max_camera_pos - min_camera_pos) + min_camera_pos
        
        qgt = np.array([0., 0, 0, 1.])
        Qgt.append(qgt) #Attenti
        
        p = np.random.normal(pgt, sigma_T, size=(3))
        T_poses[i] = ti.Vector(p)
        qnoise = np.random.uniform(qgt-sigma_Q, qgt+sigma_Q, size=4)
        q_poses[i] = ti.Vector(qnoise)
        q_poses[i] = q_poses[i]/q_poses[i].norm()

        Pgt.append(pgt)
    Pgt = np.array(Pgt)
    Qgt = np.array(Qgt)

def generate_landmarks():
    global Lgt
    Lgt = np.random.uniform(min_landmark_pos, max_landmark_pos, size=(Nl, 3))
    Lnoise = Lgt + np.random.normal(0, sigma_L, size=(Nl, 3))
    for i in range(Nl):
        L_p[i] = ti.Vector(Lnoise[i])

def generate_observations():
    c = 0
    for i in range(Nl):
        c_l = 0
        for j in range(Np):
            Pl = Lgt[i]
            p = Pgt[j]
            q = Qgt[j]
            q_inv = QuaternionInverse_(q)
            Pl = QuaternionRotate_(q_inv, Pl - p)
            x = np.random.normal(Pl[0] / Pl[2], sigma_z)
            y = np.random.normal(Pl[1] / Pl[2], sigma_z)
            if np.fabs(x) < 2 and np.fabs(y) < 2 and Pl[2] > 0:
                Z_l[c] = ti.Vector([x, y])
                Z_lind[c] = i
                Z_pind[c] = j
                c += 1
                c_l+=1
        # print("Landmark", i, Lgt[i],  "observed by", c_l)
    Nz[None] = c
    print("Observations", Nz[None])

def visual_sparse_map(qs, Ts, Ls, Lgt, ax=None, title="SparseMap", axis_len = 0.2):
    if ax is None:
        fig = plt.figure("SparseMap", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        plt.tight_layout()
    ax.clear()
    ax.set_title(title)
    # ax.scatter(Ts[:,0], Ts[:,1], Ts[:,2], color="black",marker=".",label=f"Camera Poses")
    ax.scatter(Ls[:,0], Ls[:,1], Ls[:,2], color="orange",marker="x",label=f"Landmark Est")
    ax.scatter(Lgt[:,0], Lgt[:,1], Lgt[:,2], color="gray",marker="x",label=f"Landmark GT")
    
    range_x = np.max(Ls[:,0]) - np.min(Ls[:,0])
    range_y = np.max(Ls[:,1]) - np.min(Ls[:,1])
    range_z = np.max(Ls[:,2]) - np.min(Ls[:,2])
    # ax.set_box_aspect((1, range_y/range_x, range_z/range_x))

    x_axis = np.zeros(Ts.shape)
    y_axis = np.zeros(Ts.shape)
    z_axis = np.zeros(Ts.shape)
    for i in range(len(qs)):
        x_axis[i] = QuaternionRotate_(qs[i], [1., 0., 0.])*axis_len
        y_axis[i] = QuaternionRotate_(qs[i], [0., 1., 0.])*axis_len
        z_axis[i] = QuaternionRotate_(qs[i], [0., 0., 1.])*axis_len

    ax.quiver(Ts[:,0], Ts[:,1], Ts[:,2], x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], linewidths=2, arrow_length_ratio=0.,  color="red", label="x axes")
    ax.quiver(Ts[:,0], Ts[:,1], Ts[:,2], y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], linewidths=2, arrow_length_ratio=0.,  color="green", label="y axes")
    ax.quiver(Ts[:,0], Ts[:,1], Ts[:,2], z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], linewidths=2, arrow_length_ratio=0.,  color="blue", label="z axes")

    ax.quiver(Pgt[:,0], Pgt[:,1], Pgt[:,2], Ts[:,0]-Pgt[:,0], Ts[:,1]-Pgt[:,1], Ts[:,2]-Pgt[:,2], linewidths=1, arrow_length_ratio=0., label="Camera Error")
    ax.quiver(Ls[:,0], Ls[:,1], Ls[:,2], Lgt[:,0]-Ls[:,0], Lgt[:,1]-Ls[:,1], Lgt[:,2]-Ls[:,2], linewidths=1, arrow_length_ratio=0., label="Landmark Error")

    ax.legend()

    return ax
    
def visualize(loss_log, Qlog, Tlog, Llog, Lgt):
    plt.figure("loss")
    plt.title("loss")
    ticks = range(len(loss_log))
    plt.semilogy(ticks, loss_log)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.pause(0.5)

    #Visuallize landmarks of poses
    ax = None
    Qlog = np.array(Qlog)
    Tlog = np.array(Tlog)
    Llog = np.array(Llog)
    for i in range(0, len(Qlog), 30):
        ax = visual_sparse_map(Qlog[i,:], Tlog[i,:], Llog[i,:], Lgt, ax,  title=f"TaichiSlam: SparseMap {i}")
        plt.pause(0.02)
        plt.savefig(f"output/SparseMap_{i:05d}.png")
    plt.show()

@ti.kernel
def gradient_descent():
    for i in range(Np):
        T_poses[i] -= rate*T_poses.grad[i]
        grad_tangent = q_poses.grad[i].transpose()@PlusQuaternionJacobian(q_poses[i])
        q_poses[i] = QuaternionRetraction(q_poses[i], -rate*grad_tangent.transpose()) #Need retraction
    for i in range(Nl):
        L_p[i] = L_p[i] - rate*L_p.grad[i]


def iteration():
    ti.clear_all_gradients()
    loss[None] = 0
    loss.grad[None] = 1
    func()
    func.grad()
    gradient_descent()

loss_log = []
Qlog = []
Tlog = []
Llog = []

if __name__ == "__main__":
    np.random.seed(0)
    generate_poses()
    generate_landmarks()
    generate_observations()

    loss.grad[None] = 1
    func()
    func.grad()
    print("Initial loss", loss[None])
    print("L_p.grad", L_p.grad.to_numpy())
    # print("q_poses.grad", q_poses.grad)

    for i in range(0, max_iter):
        iteration()
        loss_log.append(loss[None])
        Llog.append(L_p.to_numpy())
        Tlog.append(T_poses.to_numpy())
        Qlog.append(q_poses.to_numpy())

    visualize(loss_log, Qlog, Tlog, Llog, Lgt)
    