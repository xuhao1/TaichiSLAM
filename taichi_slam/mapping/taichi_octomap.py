# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
from .mapping_common import *
import time

@ti.data_oriented
class Octomap(Basemap):
    #If K>2 will be K**3 tree
    def __init__(self, map_scale=[10, 10], grid_scale=0.05, min_occupy_thres=3, texture_enabled=False, max_disp_particles=1000000, K=2):
        Rxy = math.ceil(math.log2(map_scale[0]/grid_scale)/math.log2(K))
        Rz = math.ceil(math.log2(map_scale[1]/grid_scale)/math.log2(K))
        self.Rxy = Rxy
        self.Rz = Rz
        self.map_scale_xy = map_scale[0]
        self.map_scale_z = map_scale[1]
        self.K = K
        self.N = self.K**self.Rxy
        self.Nz = self.K**self.Rz
        self.grid_scale_xy = self.map_scale_xy/self.N
        self.grid_scale_z = self.map_scale_z/self.Nz
        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres

        self.TEXTURE_ENABLED = texture_enabled

        self.initialize_fields()
        self.construct_octo_tree()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.i32, shape=self.max_disp_particles)

        self.grid_scale_ = ti.Vector([self.grid_scale_xy, self.grid_scale_xy, self.grid_scale_z])
        self.map_scale_ = ti.Vector([self.map_scale_xy, self.map_scale_xy, self.map_scale_z])
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2])
        self.N_ = ti.Vector([self.N, self.N, self.Nz])


    def construct_octo_tree(self):
        K = self.K
        B = ti.root
        
        for r in range(self.Rxy):
            if r < self.Rz:
                B = B.pointer(ti.ijk, (K, K, K))
            else:
                B = B.pointer(ti.ijk, (K, K, 1))
        self.B = B
        #qt.parent is the deepest of bitmasked
        self.occupy = ti.field(ti.i32)
        self.B.place(self.occupy)

        if self.TEXTURE_ENABLED:
            self.color = ti.Vector.field(3, ti.i32)
            self.B.place(self.color)

        print(f'The map voxel is:[{self.N}x{self.N}x{self.Nz}] all {self.N*self.N*self.Nz/1024/1024:.2e}M ', end ="")
        print(f'grid scale [{self.grid_scale_xy:3.3f}x{self.grid_scale_xy:3.3f}x{self.grid_scale_z:3.3f}] ', end="")
        print(f'map scale:[{self.map_scale_xy}mx{self.map_scale_xy}mx{self.map_scale_z}m] ', end ="")
        print(f'tree depth [{self.Rxy}, {self.Rz}]')


    @ti.kernel
    def get_voxel_to_particles(self, level: ti.template()):
        # Number for level
        self.num_export_particles[None] = 0
        tree = ti.static(self.occupy)
        #TODO:Each leaf needs to compute its own leaf's occupy
        if ti.static(level) > 0:
            tree = ti.static(self.occupy.parent(level))
        for i, j, k in tree:
            if self.occupy[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_disp_particles:
                    for d in ti.static(range(3)):
                        self.export_x[index][d] = ti.static([i, j, k][d])*self.grid_scale_[d] - self.map_scale_[d]/2
                        if ti.static(self.TEXTURE_ENABLED):
                            self.export_color[index] = self.color[i, j, k]
                else:
                    return


    @ti.kernel
    def recast_pcl_to_map(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])
            pt = self.input_R@pt + self.input_T
            pt = pt / self.grid_scale_ + self.NC_
            pti = pt.cast(int)

            for d in ti.static(range(3)):
                if pti[d] >= self.N_[d]:
                    pti[d] = self.N_[d] - 1
                if pti[d] < 0:
                    pti[d] = 0

            self.occupy[pti] += 1

            if ti.static(self.TEXTURE_ENABLED):
                for d in ti.static(range(3)):
                    self.color[pti][d] = rgb_array[index, d]

    def get_output_particles(self):
        return self.export_x.to_numpy(), self.export_color.to_numpy()

    def handle_render(self, scene, gui, pars, level, substeps = 3, pars_sdf=None):
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == "-":
                level += 1
                if level == self.Rxy:
                    level = self.Rxy - 1
            elif e.key == "=":
                level -= 1
                if level < 0:
                    level = 0
        t_v2p = time.time()
        self.get_voxel_to_particles(level)
        pos_, color_ = self.get_output_particles()
        t_v2p = ( time.time() - t_v2p)*1000

        cur_grid_size = (self.K**(level))*self.grid_scale_xy
        self.render_occupy_map_to_particles(pars, pos_, color_/255.0, self.num_export_particles[None], cur_grid_size)

        for i in range(substeps):
            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.text(content=f'Level {level:.2f} num_particles {self.num_export_particles[None]} grid_scale {cur_grid_size} incress =; decress -',
                    pos=(0, 0.8),
                    font_size=20,
                    color=(0x0808FF))
            gui.show()
        return level, t_v2p