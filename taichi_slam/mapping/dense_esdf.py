# This file is an easy voxblox implentation based on taichi lang

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from .mapping_common import *

@ti.data_oriented
class DenseESDF(Basemap):
    def __init__(self, map_scale=[10, 10], grid_scale=0.05, min_occupy_thres=3, texture_enabled=False, max_disp_particles=1000000, block_size=16):
        self.map_scale_xy = map_scale[0]
        self.map_scale_z = map_scale[1]

        self.block_size = block_size
        self.N = math.ceil(map_scale[0] / grid_scale/block_size)*block_size
        self.Nz = math.ceil(map_scale[1] / grid_scale/block_size)*block_size

        self.block_num_xy = math.ceil(map_scale[0] / grid_scale/block_size)
        self.block_num_z = math.ceil(map_scale[1] / grid_scale/block_size)

        self.grid_scale_xy = self.map_scale_xy/self.N
        self.grid_scale_z = self.map_scale_z/self.Nz
        
        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres

        self.TEXTURE_ENABLED = texture_enabled

        self.initialize_fields()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_TSDF_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_ESDF_particles = ti.field(dtype=ti.i32, shape=())
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.i32, shape=self.max_disp_particles)
        self.export_TSDF = ti.field(dtype=ti.i32, shape=self.max_disp_particles)
        self.export_ESDF = ti.field(dtype=ti.i32, shape=self.max_disp_particles)

        self.grid_scale_ = ti.Vector([self.grid_scale_xy, self.grid_scale_xy, self.grid_scale_z])
        self.map_scale_ = ti.Vector([self.map_scale_xy, self.map_scale_xy, self.map_scale_z])
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2])
        self.N_ = ti.Vector([self.N, self.N, self.Nz])

        self.occupy = ti.field(dtype=int)
        self.TSDF = ti.field(dtype=int)
        self.W_TSDF = ti.field(dtype=int)
        self.ESDF = ti.field(dtype=int)
        self.color = ti.Vector.field(3, ti.i32)

        block_num_xy = self.block_num_xy
        block_num_z = self.block_num_z
        block_size = self.block_size

        B = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
        B = B.dense(ti.ijk, (block_size, block_size, block_size))
        B.place(self.occupy, self.W_TSDF,self.TSDF, self.ESDF)
        
        C = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
        C = C.dense(ti.ijk, (block_size, block_size, block_size))
        C.place(self.color)
        
        self.B = B
        self.C = C

    @ti.kernel
    def recast_pcl_to_map(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])

            pt = self.input_R@pt + self.input_T
            pt = pt / self.grid_scale_ + self.NC_
            pt.cast(int)

            for d in ti.static(range(3)):
                if pt[d] >= self.N_[d]:
                    pt[d] = self.N_[d] - 1
                if pt[d] < 0:
                    pt[d] = 0

            self.occupy[pt] += 1

            if ti.static(self.TEXTURE_ENABLED):
                for d in ti.static(range(3)):
                    self.color[pt][d] = rgb_array[index, d]
    
    def get_output_particles(self):
        return self.export_x.to_numpy(),self.TSDF.to_numpy(), self.ESDF.to_numpy(), self.export_color.to_numpy()

    @ti.kernel
    def get_occupy_to_particles(self, level: ti.template()):
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
    def get_TSDF_to_particles(self, level: ti.template()):
        # Number for TSDF
        self.num_export_TSDF_particles[None] = 0
        for i, j, k in self.TSDF:
            if self.TSDF[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_TSDF_particles[None], 1)
                if self.num_export_TSDF_particles[None] < self.max_disp_particles:
                    self.export_TSDF[index] = self.TSDF[i, j, k]
                else:
                    return

    @ti.kernel
    def get_ESDF_to_particles(self, level: ti.template()):
        # Number for ESDF
        self.num_export_ESDF_particles[None] = 0
        for i, j, k in self.ESDF:
            if self.ESDF[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_ESDF_particles[None], 1)
                if self.num_export_ESDF_particles[None] < self.max_disp_particles:
                    self.export_ESDF[index] = self.ESDF[i, j, k]
                else:
                    return
    
    def handle_render(self, scene, gui, pars, level, substeps = 3):
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
        self.get_occupy_to_particles(level)
        pos_, tsdf_, esdf_, color_ = self.get_output_particles()
        self.render_occupy_map_to_particles(pars, pos_, color_/255.0, self.num_export_particles[None], self.grid_scale_xy)

        for i in range(substeps):
            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.text(content=f'Level {level:.2f} num_particles {self.num_export_particles[None]} grid_scale {self.grid_scale_xy} incress =; decress -',
                    pos=(0, 0.8),
                    font_size=20,
                    color=(0x0808FF))

            gui.show()
        return level, pos_