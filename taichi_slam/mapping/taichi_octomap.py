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
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=3, texture_enabled=False, 
            min_ray_length=0.3, max_ray_length=3.0, max_disp_particles=1000000, K=2):
        super(Octomap, self).__init__(voxel_size)
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
        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length

        self.enable_texture = texture_enabled

        self.initialize_fields()
        self.construct_octo_tree()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.export_x = ti.Vector.field(3, ti.f32, self.max_disp_particles)
        self.export_color = ti.Vector.field(3, ti.f32, self.max_disp_particles)

        self.map_size_ = ti.Vector([self.map_size_xy, self.map_size_xy, self.map_size_z])
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

        offset = [-self.N//2, -self.N//2, -self.Nz//2]
        self.B = B
        #qt.parent is the deepest of bitmasked
        self.occupy = ti.field(ti.i32)
        self.B.place(self.occupy, offset=offset)
        if self.enable_texture:
            self.color = ti.Vector.field(3, ti.f32)
            self.B.place(self.color, offset=offset)

        print(f'The map voxel is:[{self.N}x{self.N}x{self.Nz}] all {self.N*self.N*self.Nz/1024/1024:.2e}M ', end ="")
        print(f'grid scale [{self.voxel_size_xy:3.3f}x{self.voxel_size_xy:3.3f}x{self.voxel_size_z:3.3f}] ', end="")
        print(f'map scale:[{self.map_size_xy}mx{self.map_size_xy}mx{self.map_size_z}m] ', end ="")
        print(f'tree depth [{self.Rxy}, {self.Rz}]')

    @ti.func
    def is_occupy(self, i, j, k):
        return self.occupy[i, j, k] > self.min_occupy_thres

    @ti.kernel
    def cvt_occupy_to_voxels(self, level: ti.template()):
        # Number for level
        self.num_export_particles[None] = 0
        #tree = self.occupy.parent(level)
        for i, j, k in self.occupy.parent(level):
            if self.is_occupy(i, j, k):
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_disp_particles:
                    for d in ti.static(range(3)):
                        self.export_x[index][d] =[i, j, k][d]*self.voxel_size_[d] - self.map_size_[d]/2
                        if ti.static(self.enable_texture):
                            self.export_color[index] = self.color[i, j, k]

    @ti.func 
    def process_point(self, pt, rgb=None):
        ijk = self.xyz_to_0ijk(pt)
        self.occupy[ijk] += 1
        if ti.static(self.enable_texture):
            #Stupid OpenCV is BGR.
            self.color[ijk][0] = ti.cast(rgb[2], ti.float32)/255.0
            self.color[ijk][1] = ti.cast(rgb[1], ti.float32)/255.0
            self.color[ijk][2] = ti.cast(rgb[0], ti.float32)/255.0

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        self.set_pose(R, T)
        self.recast_pcl_to_map_kernel(xyz_array, rgb_array, n)
    
    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
        self.set_pose(R, T)
        self.recast_depth_to_map_kernel(depthmap, texture, w, h, K, Kcolor)

    @ti.kernel
    def recast_pcl_to_map_kernel(self, xyz_array: ti.types.ndarray(), rgb_array: ti.types.ndarray(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])
            pt = self.input_R[None]@pt + self.input_T[None]
            if ti.static(self.enable_texture):
                self.process_point(pt, rgb_array[index])
            else:
                self.process_point(pt)

    @ti.kernel
    def recast_depth_to_map_kernel(self, depthmap: ti.types.ndarray(), texture: ti.types.ndarray(), w: ti.i32, h: ti.i32, K:ti.types.ndarray(), Kcolor:ti.types.ndarray()):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        for j in range(h):
            for i in range(w):
                if depthmap[j, i] == 0 or depthmap[j, i] > ti.static(self.max_ray_length*1000) or depthmap[j, i] < ti.static(self.min_ray_length*1000):
                    continue
                dep = depthmap[j, i]/1000.0
                pt = ti.Vector([
                    (i-cx)*dep/fx, 
                    (j-cy)*dep/fy, 
                    dep])
                pt_map = self.input_R[None]@pt + self.input_T[None]
                if ti.static(self.enable_texture):
                    fx_c = Kcolor[0]
                    fy_c = Kcolor[4]
                    cx_c = Kcolor[2]
                    cy_c = Kcolor[5]
                    color_i = ti.cast((i-cx)/fx*fx_c+cx_c, ti.int32)
                    color_j = ti.cast((j-cy)/fy*fy_c+cy_c, ti.int32)

                    if color_i < 0 or color_i >= 640 or color_j < 0 or color_j >= 480:
                        continue

                    self.process_point(pt_map, [texture[color_j, color_i, 0], texture[color_j, color_i, 1], texture[color_j, color_i, 2]])
                else:
                    self.process_point(pt_map)
                    
    def get_occupy_voxels(self, l):
        self.cvt_occupy_to_voxels(l)
        return self.export_x.to_numpy(), self.export_color.to_numpy()