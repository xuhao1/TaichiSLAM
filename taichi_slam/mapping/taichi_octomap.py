# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import tina
import time
from matplotlib import cm

@ti.data_oriented
class Octomap:
    K = 2
    def __init__(self):
        self.R = 8
        self.Rz = 7
        self.N = Octomap.K**self.R
        self.Nz = Octomap.K**self.Rz
        self.map_scale_xy = 20
        self.map_scale_z = 10
        self.grid_scale_xy = self.map_scale_xy/self.N
        self.grid_scale_z = self.map_scale_z/self.Nz
        self.max_num_particles = 100000
        self.MIN_RECAST_THRES = 2

        self.TEXTURE_ENABLED = False

        self.initialize_fields()
        self.construct_octo_tree()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_num_particles)
        self.color = ti.Vector.field(3, dtype=ti.i32, shape=self.max_num_particles)

        self.grid_scale_ = ti.Vector([self.grid_scale_xy, self.grid_scale_xy, self.grid_scale_z])
        self.map_scale_ = ti.Vector([self.map_scale_xy, self.map_scale_xy, self.map_scale_z])
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2])
        self.N_ = ti.Vector([self.N, self.N, self.N])


    def construct_octo_tree(self):
        K = Octomap.K
        B = ti.root
        
        for r in range(self.R):
            if r < self.Rz:
                B = B.pointer(ti.ijk, (K, K, K))
            else:
                B = B.pointer(ti.ijk, (K, K, 1))
        self.B = B
        C = ti.root
        for r in range(self.R):
            if r < self.Rz:
                C = C.pointer(ti.ijk, (K, K, K))
            else:
                C = C.pointer(ti.ijk, (K, K, 1))
        self.C = C
        
        #qt.parent is the deepest of bitmasked
        self.qt = ti.field(ti.i32)
        self.Cqt = ti.Vector.field(3, ti.i32)

        self.B.place(self.qt)
        self.C.place(self.Cqt)
        
        self.B.deactivate_all()
        self.C.deactivate_all()

        print(f'The map voxel is:[{self.N}x{self.N}x{self.Nz}]', end ="")
        print(f'map scale:[{self.map_scale_xy}mx{self.map_scale_xy}mx{self.map_scale_z}m]', end ="")
        print(f'grid scale [{self.grid_scale_xy:3.3f}x{self.grid_scale_xy:3.3f}x{self.grid_scale_z:3.3f}]')


    @ti.kernel
    def get_voxel_to_particles(self, level: ti.template()):
        # Number for level
        self.num_export_particles[None] = 0
        tree = ti.static(self.qt)
        if ti.static(level) > 0:
            tree = ti.static(self.qt.parent(level))
        for i, j, k in tree:
            if self.qt[i, j, k] > self.MIN_RECAST_THRES:
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_num_particles:
                    for d in ti.ti.static(range(3)):
                        self.x[index][d] = ti.static([i, j, k][d])*self.grid_scale_[d] - self.map_scale_[d]/2
                        if self.TEXTURE_ENABLED:
                            self.color[index] = self.Cqt[i, j, k]


    @ti.kernel
    def recast_pcl_to_map(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32, no_project: ti.template()):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])
            if not ti.static(no_project):
                pt = self.input_R@pt + self.input_T
            
            pt = pt / self.grid_scale_ + self.NC_
            pt.cast(int)

            for d in ti.ti.static(range(3)):
                if pt[d] >= self.N_[d]:
                    pt[d] = self.N_[d] - 1
                if pt[d] < 0:
                    pt[d] = 0

            self.qt[pt] += 1

            if ti.static(self.TEXTURE_ENABLED):
                for d in ti.ti.static(range(3)):
                    self.Cqt[pt][d] = rgb_array[index, d]

    def render_map_to_particles(self, pars, pos_, colors, num_particles_, level):
        if num_particles_ == 0:
            return
        pos = pos_[0:num_particles_,:]
        if not self.TEXTURE_ENABLED:
            max_z = np.max(pos[:,2])
            min_z = np.min(pos[:,2])
            colors = cm.jet((pos[:,2] - min_z)/(max_z-min_z))
        pars.set_particles(pos)
        radius = np.ones(num_particles_)*(self.K**(level-1))*self.grid_scale_xy
        pars.set_particle_radii(radius)
        pars.set_particle_colors(colors)


    @ti.kernel
    def random_init_octo(self, pts: ti.template()):
        for i in range(pts):
            x_ = ti.random(dtype = int)%self.N
            y_ = ti.random(dtype = int)%self.N
            z_ = ti.random(dtype = int)%self.Nz
            self.qt[x_, y_, z_] =  ti.random(dtype = int)%10

def handle_render(octomap, scene, gui, pars, level, substeps = 3):
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == "-":
            level += 1
            if level == octomap.R:
                level = octomap.R - 1
        elif e.key == "=":
            level -= 1
            if level < 0:
                level = 0
    octomap.get_voxel_to_particles(level)
    pos_ = octomap.x.to_numpy()
    color_ = octomap.color.to_numpy()
    octomap.render_map_to_particles(pars, pos_, color_, octomap.num_export_particles[None], level)

    for i in range(substeps):
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.text(content=f'Level {level:.2f} num_particles {octomap.num_export_particles[None]} grid_scale {(octomap.K**(level))*octomap.grid_scale_xy} incress =; decress -',
                pos=(0, 0.8),
                font_size=20,
                color=0x080808)

        gui.show()
    return level, pos_

if __name__ == '__main__':
    RES_X = 1024
    RES_Y = 768
    gui = ti.GUI('TaichiOctomap', (RES_X, RES_Y))
    level = 2
    scene = tina.Scene(RES_X, RES_Y,  bgcolor=0xDDDDDD)
    pars = tina.SimpleParticles()
    material = tina.Lamp()
    scene.add_object(pars, material)
    octomap = Octomap()
    scene.init_control(gui, radius=octomap.map_scale_xy*2, theta=-1.0, center=(0, 0, 0), is_ortho=True)

    #Level = 0 most detailed
    octomap.random_init_octo(1000)

    while gui.running:
        level, _ = handle_render(octomap, scene, gui, pars, level)