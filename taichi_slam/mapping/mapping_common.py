import taichi as ti
import numpy as np
from matplotlib import cm

@ti.func
def sign(val):
    return (0 < val) - (val < 0)

@ti.data_oriented
class Basemap:
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
    
    def set_pose(self, _R, _T):
        for i in range(3):
            self.input_T[None][i] = _T[i]
            for j in range(3):
                self.input_R[None][i, j] = _R[i, j]