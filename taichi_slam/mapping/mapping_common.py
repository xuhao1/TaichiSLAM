import taichi as ti
import numpy as np
from matplotlib import cm

@ti.func
def sign(val):
    return (0 < val) - (val < 0)

@ti.data_oriented
class Basemap:
    def __init__(self):
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.base_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.base_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.base_T_np = np.zeros(3)
        self.base_R_np = np.eye(3)
        self.initialize_base_fields()
        self.frame_id = 0
    
    @ti.kernel
    def initialize_base_fields(self):
        self.input_R[None] = ti.Matrix.identity(ti.f32, 3)
        self.input_T[None] = ti.Matrix.zero(ti.f32, 3)
        self.base_R[None] = ti.Matrix.identity(ti.f32, 3)
        self.base_T[None] = ti.Matrix.zero(ti.f32, 3)

    @ti.kernel
    def random_init_octo(self, pts: ti.template()):
        for i in range(pts):
            x_ = ti.random(dtype = int)%self.N
            y_ = ti.random(dtype = int)%self.N
            z_ = ti.random(dtype = int)%self.Nz
            self.occupy[x_, y_, z_] =  ti.random(dtype = int)%10

    def render_map_to_particles(self, pars, pos_, colors, num_particles_, level):
        pass

    def render_occupy_map_to_particles(self, pars, pos_, colors, num_particles_, voxel_size):
        if num_particles_ == 0:
            return
        pos = pos_[0:num_particles_,:]
        if not self.TEXTURE_ENABLED:
            max_z = np.max(pos[0:num_particles_,2])
            min_z = np.min(pos[0:num_particles_,2])
            colors = cm.jet((pos[0:num_particles_,2] - min_z)/(max_z-min_z))
        pars.set_particles(pos)
        radius = np.ones(num_particles_)*voxel_size/2
        pars.set_particle_radii(radius)
        pars.set_particle_colors(colors)
    
    def convert_by_base(self, R, T):
        base_R_inv = self.base_R_np.T
        R_ = base_R_inv @ R
        T_ = base_R_inv @ (T - self.base_T_np)
        return R_, T_
    
    def set_base_pose(self, _R, _T):
        self.base_T_np = _T
        self.base_R_np = _R
        for i in range(3):
            self.base_T[None][i] = _T[i]
            for j in range(3):
                self.base_R[None][i, j] = _R[i, j]

    def set_pose(self, _R, _T):
        _R, _T = self.convert_by_base(_R, _T)
        for i in range(3):
            self.input_T[None][i] = _T[i]
            for j in range(3):
                self.input_R[None][i, j] = _R[i, j]