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
class Octomap(BaseMap):
    #If K>2 will be K**3 tree
    def __init__(self, map_scale=[10, 10], voxel_scale=0.05, min_occupy_thres=3, texture_enabled=False, 
            min_ray_length=0.3, max_ray_length=3.0, max_disp_particles=1000000, K=2, 
            max_submap_num=1024, disp_ceiling=10.0, disp_floor = -10.0, 
            is_global_map=False, recast_step=2, color_same_proj=True):
        super(Octomap, self).__init__(voxel_scale)
        Rxy = math.ceil(math.log2(map_scale[0]/voxel_scale)/math.log2(K))
        Rz = math.ceil(math.log2(map_scale[1]/voxel_scale)/math.log2(K))
        self.Rxy = Rxy
        self.Rz = Rz
        self.map_size_xy = map_scale[0]
        self.map_size_z = map_scale[1]
        self.K = K
        self.N = self.K**self.Rxy
        self.Nz = self.K**self.Rz
        self.voxel_scale = self.map_size_xy/self.N
        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres
        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length

        self.enable_texture = texture_enabled
        self.max_submap_num = max_submap_num
        self.disp_ceiling = disp_ceiling
        self.disp_floor = disp_floor
        self.is_global_map = is_global_map
        self.recast_step = recast_step
        self.color_same_proj = color_same_proj

        self.initialize_fields()
        self.construct_octo_tree()
        self.initialize_submap_fields(self.max_submap_num)


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
        B = self.root = ti.root.pointer(ti.ijkl, (self.max_submap_num, K, K, K))
        for r in range(self.Rxy):
            if r < self.Rz:
                B = B.pointer(ti.ijkl, (1, K, K, K))
            else:
                B = B.pointer(ti.ijkl, (1, K, K, 1))

        offset = [0, -self.N//2, -self.N//2, -self.Nz//2]
        self.B = B
        #qt.parent is the deepest of bitmasked
        self.occupy = ti.field(ti.f32)
        self.B.place(self.occupy, offset=offset)
        if self.enable_texture:
            self.color = ti.Vector.field(3, ti.f32)
            self.B.place(self.color, offset=offset)

        print(f'The map voxel is:[{self.max_submap_num}x{self.N}x{self.N}x{self.Nz}] all {self.N*self.N*self.Nz/1024/1024:.2e}M ', end ="")
        print(f'voxel scale {self.voxel_scale:3.3f}^3 ', end="")
        print(f'map scale:[{self.map_size_xy}mx{self.map_size_xy}mx{self.map_size_z}m] ', end ="")
        print(f'tree depth [{self.Rxy}, {self.Rz}]')

    @ti.func
    def is_occupy(self, sijk):
        return self.occupy[sijk] > self.min_occupy_thres

    @ti.kernel
    def cvt_occupy_to_voxels(self, level: ti.template()):
        # Number for level
        self.num_export_particles[None] = 0
        #tree = self.occupy.parent(level)
        for sijk in ti.grouped(self.occupy.parent(level)):
            if sijk[0] == self.active_submap_id[None]:
                if self.is_occupy(sijk):
                    index = ti.atomic_add(self.num_export_particles[None], 1)
                    if self.num_export_particles[None] < self.max_disp_particles:
                        self.export_x[index] = self.sijk_to_xyz(sijk)
                        if ti.static(self.enable_texture):
                            self.export_color[index] = self.color[sijk]
    
    @ti.kernel
    def cvt_occupy_voxels_to(self, level: ti.template(), cur_num: ti.template(), 
            max_disp_particles:ti.template(), x: ti.template(), color: ti.template()):
        for sijk in ti.grouped(self.occupy.parent(level)):
            if sijk[0] == self.active_submap_id[None]:
                if self.is_occupy(sijk):
                    index = ti.atomic_add(cur_num[None], 1)
                    if cur_num[None] < max_disp_particles:
                        x[index] = self.sijk_to_xyz(sijk)
                        if ti.static(self.enable_texture):
                            color[index] = self.color[sijk]

    @ti.func 
    def process_point(self, pt, rgb=None):
        ijk = self.xyz_to_sijk(pt)
        self.occupy[ijk] += 1
        if ti.static(self.enable_texture):
            #Stupid OpenCV is BGR.
            self.color[ijk][0] = ti.cast(rgb[2], ti.float32)/255.0
            self.color[ijk][1] = ti.cast(rgb[1], ti.float32)/255.0
            self.color[ijk][2] = ti.cast(rgb[0], ti.float32)/255.0

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        self.set_pose(R, T)
        self.recast_pcl_to_map_kernel(xyz_array, rgb_array, n)
    
    def recast_depth_to_map(self, R, T, depthmap, texture):
        self.set_pose(R, T)
        self.recast_depth_to_map_kernel(depthmap, texture)

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
    def recast_depth_to_map_kernel(self, depthmap: ti.types.ndarray(), texture: ti.types.ndarray()):
        h = depthmap.shape[0]
        w = depthmap.shape[1]
        for jj in range(0, h/ti.static(self.recast_step)):
            j = jj*ti.static(self.recast_step)
            for ii in range(0, w/ti.static(self.recast_step)):
                i = ii*ti.static(self.recast_step)
                if depthmap[j, i] == 0 or depthmap[j, i] > ti.static(self.max_ray_length*1000) or depthmap[j, i] < ti.static(self.min_ray_length*1000):
                    continue
                dep = depthmap[j, i]/1000.0
                pt = self.unproject_point_dep(i, j, dep)
                pt_map = self.input_R[None]@pt + self.input_T[None]
                if ti.static(self.enable_texture):
                    if ti.static(self.color_same_proj):
                        color = [texture[j, i, 0], texture[j, i, 1], texture[j, i, 2]]
                        self.process_point(pt_map, color)
                    else:
                        colorij = self.color_ind_from_depth_pt(i, j, texture.shape[1], texture.shape[0])
                        color = [texture[colorij, 0], texture[colorij, 1], texture[colorij, 2]]
                        self.process_point(pt_map, color)
                else:
                    self.process_point(pt_map)
    
    @ti.kernel
    def fuse_submaps_kernel(self, num_submaps: ti.i32, submap_occupy: ti.template(), submap_color: ti.template(),
            submaps_base_R_np: ti.types.ndarray(), submaps_base_T_np: ti.types.ndarray()):
        # self.set_base_poses_submap(num_submaps, submaps_base_R_np, submaps_base_T_np)
        for s, i, j, k in submap_occupy:
            submap_occ = submap_occupy[s, i, j, k]
            if submap_occ > self.min_occupy_thres:
                xyz = self.submap_i_j_k_to_xyz(s, i, j, k)
                ijk =  ti.round(xyz / self.voxel_scale_, ti.i32)
                ijk_ = ti.Vector([0, ijk[0], ijk[1], ijk[2]], ti.i32)
                occ = self.occupy[ijk_]
                self.occupy[ijk_] += submap_occ
                if ti.static(self.enable_texture):
                    self.color[ijk_] = (occ*self.color[ijk_] + submap_occ*submap_color[s, i, j, k])/self.occupy[ijk_] 
                
    def get_occupy_voxels(self, l):
        self.cvt_occupy_to_voxels(l)
        return self.export_x.to_numpy(), self.export_color.to_numpy()

    def fuse_submaps(self, submaps):
        self.reset()
        t = time.time()
        self.fuse_submaps_kernel(submaps.active_submap_id[None], submaps.occupy, submaps.color, self.submaps_base_R_np, self.submaps_base_T_np)
        print(f"[OctoMap] Fuse submaps {(time.time() - t)*1000:.1f}ms, active local: {submaps.active_submap_id[None]} remote: {submaps.remote_submap_num[None]}")

    def saveMap(self, path):
        pass

    def export_submap(self):
        return {}
    
    def finalization_current_submap(self):
        pass

    def reset(self):
        self.root.deactivate_all()