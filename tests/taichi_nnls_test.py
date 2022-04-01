#!/usr/bin/env python3
import os, sys
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.utils.visualization import *
import time
from taichi_slam.taichi_opti.taichi_nnls import *
from taichi_slam.taichi_transformations import *


@ti.func
def project_to_image(T):
    return ti.Vector([T[0]/T[2], T[1]/T[2]])

class ReprojectionError(CostFunction):
    def __init__(self, z_obs):
        self.z_obs = z_obs
        self.sqrt_inf_z = 1.
        
    @ti.func
    def reprojection_err(self, z_l, l_p, q, T):
        l_ = l_p - T
        l_b = QuaternionRotate(QuaternionInverse(q), l_)
        z = project_to_image(l_b)
        e = (z - z_l)*self.sqrt_inf_z
        return ti.Vector([e[0], e[1]])

    @ti.func
    def retrive_param(self, field, index):
        print(index)
        s = self.indices[index, 0]
        e = self.indices[index, 0] + self.indices[index, 1] 
        x = ti.Vector([0. for i in range(MAX_PARAM_DIM)])
        x[0] = field[s]
        x[1] = field[s+1]
        x[2] = field[s+2]
        x[3] = field[s+3]
        return x

    @ti.func
    def evaluate(self, field):
        q = self.retrive_param(field, 0)
        T = self.retrive_param(field, 1)[0:3]
        l_p = self.retrive_param(field, 2)[0:3]

        print("q", q, "T", T, "l_p", l_p)
        return self.reprojection_err(self.z_obs, l_p, q, T)
    
    def residual_dim(self):
        return 2


if __name__ == "__main__":
    ti.init(ti.cpu, dynamic_index=True)
    err = ReprojectionError([0., 0.])
    q = np.array([0., 0., 0., 1.])
    T = np.array([0.1, 0.2, -0.1, 0.])
    L_init = np.array([1., 1., 1., 0.])

    nnls = TaichiNNLS(verbose=True)
    nnls.add_cost_function(err, q, T, L_init)
    nnls.pre_solve()
    nnls.evaluate_test()