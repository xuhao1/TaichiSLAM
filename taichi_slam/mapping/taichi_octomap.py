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
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=3, texture_enabled=False, recast_max_distance=3.0, max_disp_particles=1000000, K=2):
        Rxy = math.ceil(math.log2(map_scale[0]/voxel_size)/math.log2(K))
        Rz = math.ceil(math.log2(map_scale[1]/voxel_size)/math.log2(K))
        self.Rxy = Rxy
        self.Rz = Rz
        self.map_size_xy = map_scale[0]
        self.map_size_z = map_scale[1]
        self.K = K
        self.N = self.K**self.Rxy
        self.Nz = self.K**self.Rz
        self.voxel_size_xy = self.map_size_xy/self.N
        self.voxel_size_z = self.map_size_z/self.Nz
        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres
        self.recast_max_distance = recast_max_distance

        self.TEXTURE_ENABLED = texture_enabled

        self.initialize_fields()
        self.construct_octo_tree()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        self.export_x = ti.Vector.field(3, ti.f32, self.max_disp_particles)
        self.export_color = ti.Vector.field(3, ti.f32, self.max_disp_particles)

        self.voxel_size_ = ti.Vector([self.voxel_size_xy, self.voxel_size_xy, self.voxel_size_z])
        self.map_size_ = ti.Vector([self.map_size_xy, self.map_size_xy, self.map_size_z])
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2])
        self.N_ = ti.Vector([self.N, self.N, self.Nz])
        
        self.init_fields()
    
    @ti.kernel
    def init_fields(self):
        for i in range(self.max_disp_particles):
            self.export_color[i] = ti.Vector([0.5, 0.5, 0.5])
            self.export_x[i] = ti.Vector([-100000, -100000, -100000])
            # self.export_x[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

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
        print(f'grid scale [{self.voxel_size_xy:3.3f}x{self.voxel_size_xy:3.3f}x{self.voxel_size_z:3.3f}] ', end="")
        print(f'map scale:[{self.map_size_xy}mx{self.map_size_xy}mx{self.map_size_z}m] ', end ="")
        print(f'tree depth [{self.Rxy}, {self.Rz}]')


    @ti.kernel
    def cvt_occupy_to_voxels(self, level: ti.template()):
        # Number for level
        self.num_export_particles[None] = 0
        #tree = self.occupy.parent(level)

        for i, j, k in self.occupy.parent(level):
            if self.occupy[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_disp_particles:
                    for d in ti.static(range(3)):
                        self.export_x[index][d] =[i, j, k][d]*self.voxel_size_[d] - self.map_size_[d]/2
                        if ti.static(self.TEXTURE_ENABLED):
                            self.export_color[index] = self.color[i, j, k]

    @ti.func 
    def process_point(self, pt, rgb=None):
        pt = pt / self.voxel_size_ + self.NC_
        ijk = pt.cast(int)

        for d in ti.static(range(3)):
            if ijk[d] >= self.N_[d]:
                ijk[d] = self.N_[d] - 1
            if ijk[d] < 0:
                ijk[d] = 0

        self.occupy[ijk] += 1

        if ti.static(self.TEXTURE_ENABLED):
            for d in ti.static(range(3)):
                self.color[ijk][d] = rgb[d]

    @ti.kernel
    def recast_pcl_to_map(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])
            pt = self.input_R@pt + self.input_T
            if ti.static(self.TEXTURE_ENABLED):
                self.process_point(pt, rgb_array[index])
            else:
                self.process_point(pt)

    @ti.kernel
    def recast_depth_to_map(self, depthmap: ti.ext_arr(), texture: ti.ext_arr(), w: ti.i32, h: ti.i32, K:ti.ext_arr()):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        for j in range(h):
            for i in range(w):
                if depthmap[j, i] == 0 or depthmap[j, i]/1000 > ti.static(self.recast_max_distance):
                    continue
                dep = depthmap[j, i]/1000.0
                pt = ti.Vector([
                    (i-cx)*dep/fx, 
                    (j-cy)*dep/fy, 
                    dep])
                pt = self.input_R[None]@pt + self.input_T[None]
                if ti.static(self.TEXTURE_ENABLED):
                    self.process_point(pt, [texture[j, i, 0], texture[j, i, 1], texture[j, i, 2]])
                else:
                    self.process_point(pt)
                    
    @ti.kernel
    def recast_depth_to_map_debug(self, depthmap: ti.ext_arr(), rgb_array: ti.ext_arr(), w: ti.i32, h: ti.i32, K:ti.ext_arr()):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.num_export_particles[None] = 0

        for j in range(h):
            for i in range(w):
                if depthmap[j, i] == 0 or depthmap[j, i]/1000 > ti.static(self.recast_max_distance):
                    continue
                dep = depthmap[j, i]/1000.0
                pt = ti.Vector([
                    (i-cx)*dep/fx, 
                    (j-cy)*dep/fy, 
                    dep])
                pt = self.input_R@pt + self.input_T
                index = ti.atomic_add(self.num_export_particles[None], 1)
                self.export_x[index] = pt

    def get_occupy_voxels(self, l):
        self.cvt_occupy_to_voxels(l)
        return self.export_x.to_numpy(), self.export_color.to_numpy()

    def handle_render(self, scene, gui, pars, level, substeps = 3, pars_sdf=None):

        t_v2p = time.time()
        pos_, color_ = self.get_occupy_voxels(level)
        t_v2p = ( time.time() - t_v2p)*1000

        cur_grid_size = (self.K**(level))*self.voxel_size_xy
        self.render_occupy_map_to_particles(pars, pos_, color_/255.0, self.num_export_particles[None], cur_grid_size)

        for i in range(substeps):
            for e in gui.get_events(ti.GUI.RELEASE):
                if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    exit()
                elif e.key == "[":
                    level += 1
                    if level == self.Rxy:
                        level = self.Rxy - 1
                elif e.key == "]":
                    level -= 1
                    if level < 0:
                        level = 0
                        
            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.text(content=f'Level {level:.2f} num_particles {self.num_export_particles[None]} voxel_size {cur_grid_size} incress =; decress -',
                    pos=(0, 0.8),
                    font_size=20,
                    color=(0x0808FF))
            gui.show()
        return level, t_v2p