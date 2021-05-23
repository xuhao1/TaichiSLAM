# This file is an easy voxblox implentation based on taichi lang

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from .mapping_common import *
import time

Wmax = 1000


@ti.data_oriented
class DenseESDF(Basemap):
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=0, texture_enabled=False, max_disp_particles=1000000, block_size=16, max_ray_length=10):
        self.map_size_xy = map_scale[0]
        self.map_size_z = map_scale[1]

        self.block_size = block_size
        self.N = math.ceil(map_scale[0] / voxel_size/block_size)*block_size
        self.Nz = math.ceil(map_scale[1] / voxel_size/block_size)*block_size

        self.block_num_xy = math.ceil(map_scale[0] / voxel_size/block_size)
        self.block_num_z = math.ceil(map_scale[1] / voxel_size/block_size)

        self.voxel_size_xy = self.map_size_xy/self.N
        self.voxel_size_z = self.map_size_z/self.Nz
        
        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres

        self.TEXTURE_ENABLED = texture_enabled

        self.max_ray_length = max_ray_length
        self.tsdf_surface_thres = self.voxel_size_xy
        self.gamma = 0.01

        self.initialize_fields()

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_TSDF_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_ESDF_particles = ti.field(dtype=ti.i32, shape=())
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.i32, shape=self.max_disp_particles)
        self.export_TSDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_ESDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_ESDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        
        self.updated_TSDF = ti.Vector.field(3, dtype=ti.i32, shape=self.max_disp_particles)
        self.num_updated_TSDF = ti.field(dtype=ti.i32, shape=())

        self.voxel_size_ = ti.Vector([self.voxel_size_xy, self.voxel_size_xy, self.voxel_size_z])
        self.map_size_ = ti.Vector([self.map_size_xy, self.map_size_xy, self.map_size_z])
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2])
        self.N_ = ti.Vector([self.N, self.N, self.Nz])

        self.occupy = ti.field(dtype=ti.u16)
        self.TSDF = ti.field(dtype=ti.f32)
        self.W_TSDF = ti.field(dtype=ti.f32)
        self.ESDF = ti.field(dtype=ti.f32)
        self.observed = ti.field(dtype=ti.i8)
        self.fixed = ti.field(dtype=ti.i8)
        self.parent_dir = ti.Vector.field(3, dtype=ti.i32)
        if self.TEXTURE_ENABLED:
            self.color = ti.Vector.field(3, ti.i8)

        block_num_xy = self.block_num_xy
        block_num_z = self.block_num_z
        block_size = self.block_size

        self.raise_queue = ti.Vector.field(3, dtype=ti.i32, shape=10*self.max_disp_particles)
        self.lower_queue = ti.Vector.field(3, dtype=ti.i32, shape=10*self.max_disp_particles)
        self.num_raise_queue = ti.field(dtype=ti.i32, shape=())
        self.num_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_raise_queue = ti.field(dtype=ti.i32, shape=())

        if self.block_size > 1:
            B = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            # C = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            self.Broot = B
            # self.Croot = C
            B = B.dense(ti.ijk, (block_size, block_size, block_size))
            # C = C.dense(ti.ijk, (block_size, block_size, block_size))
            B.place(self.occupy, self.W_TSDF,self.TSDF, self.ESDF, self.observed, self.fixed, self.parent_dir)
            # C.place(self.parent_dir)
            if self.TEXTURE_ENABLED:
                B.place(self.color)
        else:
            B = ti.root.dense(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            C = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            self.Broot = B
            self.Croot = C
            B.place(self.occupy, self.W_TSDF,self.TSDF, self.ESDF, self.observed, self.fixed)
            C.place(self.parent_dir)
            if self.TEXTURE_ENABLED:
                B.place(self.color)


        self.B = B

        self.neighbors = []
        for _di in range(-1, 2):
            for _dj in range(-1, 2):
                for _dk in range(-1, 2):
                    if _di !=0 or _dj !=0 or _dk != 0:
                        self.neighbors.append(ti.Vector([_di, _dj, _dk]))
    @ti.func
    def constrain_coor(self, _i):
        _i = _i.cast(int)
        for d in ti.static(range(3)):
            if _i[d] >= self.N_[d]:
                _i[d] = self.N_[d] - 1
            if _i[d] < 0:
                _i[d] = 0
        return _i

    @ti.func 
    def w_x_p(self, d, z):
        epi = ti.static(self.voxel_size_xy)
        theta = ti.static(self.voxel_size_xy*4)
        ret = 0.0
        if d > ti.static(-epi):
            ret = 1.0/(z*z)
        elif d > ti.static(-theta):
            ret = (d+theta)/(z*z*(theta-epi))
        return ret
    
    @ti.func
    def is_fixed(self, ijk):
        return  ti.static(-self.gamma) < self.ESDF[ijk] < ti.static(self.gamma)
    
    @ti.func
    def insert_lower(self, ijk):
        if ti.is_active(self.Broot, ijk):
            #Only add activated voxel
            _index = ti.atomic_add(self.num_lower_queue[None], 1)
            self.lower_queue[_index] = ijk

    @ti.func
    def insert_raise(self, ijk):
        if ti.is_active(self.Broot, ijk):
            #Only add activated voxel
            index = ti.atomic_add(self.num_raise_queue[None], 1)
            self.raise_queue[index] = ijk

    @ti.func
    def insertneighbors_lower(self, ijk):
        for dir in ti.static(self.neighbors):
            self.insert_lower(ijk+dir)

    @ti.func
    def process_raise_queue(self):
        while self.head_raise_queue[None] < self.num_raise_queue[None]:
            _index_head = self.raise_queue[self.head_raise_queue[None]]
            self.head_raise_queue[None] += 1
            self.ESDF[_index_head] = sign(self.ESDF[_index_head]) * ti.static(self.max_ray_length)
            for dir in ti.static(self.neighbors):
                n_voxel_ijk = dir + _index_head
                if all(-dir == self.parent_dir[n_voxel_ijk]):
                    self.insert_raise(n_voxel_ijk)
                else:
                    self.insert_lower(n_voxel_ijk)

    @ti.func
    def process_lower_queue(self):
        while self.head_lower_queue[None] < self.num_lower_queue[None]:
            self.head_lower_queue[None] += 1
            _index_head = self.lower_queue[self.head_lower_queue[None]]
            if not ti.is_active(self.Broot, _index_head):
                print("Warn! Nodes in lower queue is not activate!")

            for dir in ti.static(self.neighbors):
                n_voxel_ijk = dir + _index_head
                if ti.is_active(self.Broot, n_voxel_ijk):
                    dis = ti.static(dir.norm()*self.voxel_size_xy)
                    n_esdf = self.ESDF[n_voxel_ijk] 
                    _n_esdf = self.ESDF[_index_head] + dis
                    if n_esdf > 0 and _n_esdf < n_esdf:
                        self.ESDF[n_voxel_ijk] = _n_esdf
                        self.parent_dir[n_voxel_ijk] = -dir
                        self.insert_lower(n_voxel_ijk)
                    else:
                        _n_esdf = self.ESDF[_index_head] - dis
                        if n_esdf < 0 and _n_esdf > n_esdf:
                            self.ESDF[n_voxel_ijk] = _n_esdf
                            self.parent_dir[n_voxel_ijk] = -dir
                            self.insert_lower(n_voxel_ijk)


    @ti.kernel
    def propogate_esdf(self):
        self.num_raise_queue[None] = 0
        self.num_lower_queue[None] = 0
        self.head_raise_queue[None] = 0
        self.head_lower_queue[None] = 0
        # self.parent_dir.setZero()
        for i in range(self.num_updated_TSDF[None]):
            _voxel_ijk = self.updated_TSDF[i]
            self.parent_dir[_voxel_ijk] = ti.Vector([0, 0, 0])
            t_d = self.TSDF[_voxel_ijk]
            if self.is_fixed(_voxel_ijk):
                self.ESDF[_voxel_ijk] = t_d
                if self.ESDF[_voxel_ijk] > t_d or self.observed[_voxel_ijk] != 1:
                    self.observed[_voxel_ijk] = 1
                    self.insert_lower(_voxel_ijk)
                else:
                    self.insert_raise(_voxel_ijk)
                    self.insert_lower(_voxel_ijk)
            else:
                if self.fixed[_voxel_ijk] > 0:
                    self.observed[_voxel_ijk] = 1
                    self.ESDF[_voxel_ijk] = sign(t_d)*ti.static(self.max_ray_length)
                    self.insert_raise(_voxel_ijk)
                elif self.observed[_voxel_ijk] != 1:
                    self.observed[_voxel_ijk] = 1
                    self.ESDF[_voxel_ijk] = sign(t_d)*ti.static(self.max_ray_length)
                    self.insertneighbors_lower(_voxel_ijk)

        self.process_raise_queue()
        self.process_lower_queue()

    @ti.kernel
    def recast_pcl_to_map(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
        self.num_updated_TSDF[None] = 0
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]])

            z = xyz_array[index,2]
            
            len_p2s = pt.norm()
            len_p2s_i = pt.norm()/self.voxel_size_xy

            #Vector from sensor to pointcloud, pt_s2p/pt_s2p.norm() is direction of the ray
            pt_s2p = self.input_R@pt
            d_s2p = pt_s2p /pt_s2p.norm()
            pt = pt_s2p + self.input_T

            pti = pt / self.voxel_size_ + self.NC_
            pti = self.constrain_coor(pti)
            self.occupy[pti] += 1
            #Easy recasting

            j_f = 0.0

            #Recast through the ray
            #TODO: grouped recasting
            ray_cast_voxels = ti.min(len_p2s_i, self.max_ray_length/self.voxel_size_xy)
            for j in range(1, ray_cast_voxels):
                j_f += 1.0
                x_ = d_s2p*j_f*self.voxel_size_xy + self.input_T
                xi = x_ / self.voxel_size_ + self.NC_ #index of x
                xi = self.constrain_coor(xi)

                #vector from current voxel to point, e.g. p-x
                v2p = pt - x_
                d_x_p = v2p.norm()*self.voxel_size_xy
                d_x_p_s = d_x_p*sign(v2p.dot(pt_s2p))
    
                w_x_p = self.w_x_p(d_x_p, z)

                self.TSDF[xi] =  (self.TSDF[xi]*self.W_TSDF[xi]+w_x_p*d_x_p_s)/(self.W_TSDF[xi]+w_x_p)
                self.W_TSDF[xi] = ti.min(self.W_TSDF[xi]+w_x_p, Wmax)

                _index_updated = ti.atomic_add(self.num_updated_TSDF[None], 1)
                self.updated_TSDF[_index_updated] = xi

            if ti.static(self.TEXTURE_ENABLED):
                for d in ti.static(range(3)):
                    self.color[pti][d] = rgb_array[index, d]
    
    @ti.kernel
    def cvt_occupy_to_voxels(self):
        # Number for level
        self.num_export_particles[None] = 0
        for i, j, k in self.occupy:
            if self.occupy[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_disp_particles:
                    for d in ti.static(range(3)):
                        self.export_x[index][d] = ti.static([i, j, k][d])*self.voxel_size_[d] - self.map_size_[d]/2
                        if ti.static(self.TEXTURE_ENABLED):
                            self.export_color[index] = self.color[i, j, k]
                else:
                    return

    @ti.kernel
    def cvt_TSDF_surface_to_voxels(self):
        # Number for TSDF
        self.num_export_TSDF_particles[None] = 0
        for i, j, k in self.TSDF:
            _index = (self.map_size_[2]/2)/self.voxel_size_z
            if self.occupy[i, j, k] and ti.abs(self.TSDF[i, j, k] ) < self.tsdf_surface_thres:
                index = ti.atomic_add(self.num_export_TSDF_particles[None], 1)
                if self.num_export_TSDF_particles[None] < self.max_disp_particles:
                    self.export_TSDF[index] = self.TSDF[i, j, k]
                    if ti.static(self.TEXTURE_ENABLED):
                        self.export_color[index] = self.color[i, j, k]
                    for d in ti.static(range(3)):
                        self.export_TSDF_xyz[index][d] = ti.static([i, j, k][d])*self.voxel_size_[d] - self.map_size_[d]/2
                else:
                    return

    @ti.kernel
    def cvt_ESDF_to_voxels_slice(self, z: ti.template()):
        # Number for ESDF
        self.num_export_ESDF_particles[None] = 0
        for i, j, k in self.ESDF:
            _index = (z+self.map_size_[2]/2)/self.voxel_size_z
            if _index - 0.5 < k < _index + 0.5:
                index = ti.atomic_add(self.num_export_ESDF_particles[None], 1)
                if self.num_export_ESDF_particles[None] < self.max_disp_particles:
                    self.export_ESDF[index] = self.ESDF[i, j, k]
                    for d in ti.static(range(3)):
                        self.export_ESDF_xyz[index][d] = ti.static([i, j, k][d])*self.voxel_size_[d] - self.map_size_[d]/2
                else:
                    return

    def get_voxels_occupy(self):
        self.get_occupy_to_voxels()
        return self.export_x.to_numpy(), self.export_color.to_numpy()
    
    def get_voxels_TSDF_surface(self):
        self.cvt_TSDF_surface_to_voxels()
        if self.TEXTURE_ENABLED:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), self.export_color.to_numpy()
        else:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), None
    
    def get_voxels_ESDF_slice(self, z):
        self.cvt_ESDF_to_voxels_slice(z)
        return self.export_ESDF_xyz.to_numpy(), self.export_ESDF.to_numpy(),

    def render_sdf_surface_to_particles(self, pars, pos_, sdf,  num_particles_, colors=None):
        if num_particles_ == 0:
            return
        if colors is None:
            max_z = np.max(pos_[0:num_particles_,2])
            min_z = np.min(pos_[0:num_particles_,2])
            colors = cm.gist_rainbow((pos_[0:num_particles_,2] - min_z)/(max_z-min_z))
        else:
            colors = colors/255.0

        pos = pos_[0:num_particles_,:]
        pars.set_particles(pos)
        radius = np.ones(num_particles_)*self.voxel_size_xy/2
        pars.set_particle_radii(radius)
        pars.set_particle_colors(colors)

    def render_sdf_voxel_to_particles(self, pars, pos_, sdf,  num_particles_):
        if num_particles_ == 0:
            return
        max_sdf = np.max(sdf[0:num_particles_])
        min_sdf = np.min(sdf[0:num_particles_])
        print("min_sdf", min_sdf)
        print("max_sdf", max_sdf)
        colors = cm.jet((sdf[0:num_particles_] - min_sdf)/(max_sdf-min_sdf))

        pos = pos_[0:num_particles_,:]
        pars.set_particles(pos)
        radius = np.ones(num_particles_)*self.voxel_size_xy/2
        pars.set_particle_radii(radius)
        pars.set_particle_colors(colors)


    def handle_render(self, scene, gui, pars1, level, substeps = 3, pars_sdf=None):
        t_v2p = time.time()

        tsdf_pos, tsdf, color_ = self.get_voxels_TSDF_surface()
        t_v2p = (time.time() - t_v2p)*1000
        self.render_sdf_surface_to_particles(pars1, tsdf_pos, tsdf, self.num_export_TSDF_particles[None], color_)
        
        esdf_pos, esdf = self.get_voxels_ESDF_slice(level)
        self.render_sdf_voxel_to_particles(pars_sdf, esdf_pos, esdf, self.num_export_ESDF_particles[None])

        for i in range(substeps):
            for e in gui.get_events(ti.GUI.RELEASE):
                if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    exit()
                elif e.key == "[":
                    level -= 0.2
                elif e.key == "]":
                    level += 0.2

            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.text(content=f'Level {level:.2f} num_particles {self.num_export_TSDF_particles[None]} voxel_size {self.voxel_size_xy} incress =; decress -',
                    pos=(0, 0.8),
                    font_size=20,
                    color=(0x0808FF))

            gui.show()
        return level, t_v2p