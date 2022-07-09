# This file is an easy voxblox implentation based on taichi lang

from cgitb import enable
import taichi as ti
import numpy as np
import math
from matplotlib import cm
from .mapping_common import *

Wmax = 1000

var = [1, 2, 3, 4, 5]

class SDFSubmapFields:
    def __init__(self, submap_id, enable_esdf, texture_enabled):
        self.submap_id = submap_id
        self.occupy = ti.field(dtype=ti.i32)
        self.TSDF = ti.field(dtype=ti.f32)
        self.W_TSDF = ti.field(dtype=ti.f32)
        self.TSDF_observed = ti.field(dtype=ti.i32)

        if enable_esdf:
            self.ESDF = ti.field(dtype=ti.f32)
            self.fixed = ti.field(dtype=ti.i8)
            self.parent_dir = ti.Vector.field(3, dtype=ti.i32)
            self.observed = ti.field(dtype=ti.i8)

        if texture_enabled:
            self.color = ti.Vector.field(3, dtype=ti.f32)

        self.enable_esdf = enable_esdf
        self.texture_enabled = texture_enabled

    def place(self, B):
        B.place(self.occupy, self.W_TSDF,self.TSDF, self.TSDF_observed)
        if self.enable_esdf:
            B.place(self.ESDF, self.observed, self.fixed, self.parent_dir)
        if self.texture_enabled:
            B.place(self.color)
        self.B = B

@ti.data_oriented
class DenseSDF(Basemap):
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=0, texture_enabled=False, \
            max_disp_particles=1000000, block_size=16, max_ray_length=10, min_ray_length=0.3, 
            enable_esdf=False, internal_voxels = 10, maxsubmap_num=100):
        super(DenseSDF, self).__init__()
        self.map_size_xy = map_scale[0]
        self.map_size_z = map_scale[1]

        self.block_size = block_size
        self.voxel_size = voxel_size

        self.N = math.ceil(map_scale[0] / voxel_size/block_size)*block_size
        self.Nz = math.ceil(map_scale[1] / voxel_size/block_size)*block_size

        self.block_num_xy = math.ceil(map_scale[0] / voxel_size/block_size)
        self.block_num_z = math.ceil(map_scale[1] / voxel_size/block_size)

        self.map_size_xy = voxel_size * self.N
        self.map_size_z = voxel_size * self.Nz

        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres

        self.texture_enabled = texture_enabled

        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length
        self.tsdf_surface_thres = self.voxel_size
        self.gamma = self.voxel_size
        self.enable_esdf = enable_esdf
        self.internal_voxels = internal_voxels
        self.maxsubmap_num = maxsubmap_num

        self.initialize_fields()
        self.set_active_submap(0)

    def data_structures(self, block_num_xy, block_num_z, block_size):
        if block_size > 1:
            B = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z)).dense(ti.ijk, (block_size, block_size, block_size))
        else:
            B0 = ti.root.dynamic(ti.i, self.maxsubmap_num)
            B = B0.dense(ti.ikl, (block_size, block_size, block_size))
        return B
    
    def data_structures_grouped(self, block_num_xy, block_num_z, block_size):
        if block_size > 1:
            Broot = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijk, (block_size, block_size, block_size))
        else:
            B = Broot = ti.root.dense(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
        return B, Broot

    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_TSDF_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_ESDF_particles = ti.field(dtype=ti.i32, shape=())

        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_ESDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_ESDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        
        self.voxel_size_ = ti.Vector([self.voxel_size, self.voxel_size, self.voxel_size], ti.f32)
        self.map_size_ = ti.Vector([self.map_size_xy, self.map_size_xy, self.map_size_z], ti.f32)
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2], ti.f32)
        self.N_ = ti.Vector([self.N, self.N, self.Nz], ti.f32)

        block_num_xy = self.block_num_xy
        block_num_z = self.block_num_z
        block_size = self.block_size

        self.max_queue_size = 1000000
        self.raise_queue = ti.Vector.field(3, dtype=ti.i32, shape=self.max_queue_size)
        self.lower_queue = ti.Vector.field(3, dtype=ti.i32, shape=self.max_queue_size)
        self.num_raise_queue = ti.field(dtype=ti.i32, shape=())
        self.num_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_raise_queue = ti.field(dtype=ti.i32, shape=())

        #Store the SDF fields
        self.sdf_fields = []
        for i in range(self.maxsubmap_num):
            self.sdf_fields.append(SDFSubmapFields(i, self.enable_esdf, self.texture_enabled))
            B = self.data_structures(block_num_xy, block_num_z, block_size)
            self.sdf_fields[i].place(B)

        #For grouping the sensor input
        self.new_pcl_count = ti.field(dtype=ti.i32)
        self.new_pcl_sum_pos = ti.Vector.field(3, dtype=ti.f32) #position in sensor coor
        self.new_pcl_z = ti.field(dtype=ti.f32) #position in sensor coor
        if self.enable_esdf:
            self.T = self.data_structures(block_num_xy, block_num_z, block_size)
            self.T.place(self.updated_TSDF)
        self.PCL, self.PCLroot = self.data_structures_grouped(block_num_xy, block_num_z, block_size)
        self.PCL.place(self.new_pcl_count, self.new_pcl_sum_pos, self.new_pcl_z)
        if self.texture_enabled:
            self.new_pcl_sum_color = ti.Vector.field(3, dtype=ti.f32)
            self.PCL.place(self.new_pcl_sum_color)

        self.neighbors = []
        for _di in range(-1, 2):
            for _dj in range(-1, 2):
                for _dk in range(-1, 2):
                    if _di !=0 or _dj !=0 or _dk != 0:
                        self.neighbors.append(ti.Vector([_di, _dj, _dk], ti.f32))
        
        self.colormap = ti.Vector.field(3, float, shape=1024)
        self.color_rate = 2
        for i in range(1024):
            self.colormap[i][0] = cm.bwr(i/1024.0)[0]
            self.colormap[i][1] = cm.bwr(i/1024.0)[1]
            self.colormap[i][2] = cm.bwr(i/1024.0)[2]

        self.init_fields()

    def set_active_submap(self, submap_id):
        print("DenseSDF change to submap", submap_id)
        self.active_submap = self.sdf_fields[submap_id]
        self.active_submap_id = submap_id
        submap = self.sdf_fields[submap_id]
        print("Submap TSDF", id(submap.TSDF.parent()))
        self.occupy = submap.occupy
        self.TSDF = submap.TSDF
        self.W_TSDF = submap.W_TSDF
        self.TSDF_observed = submap.TSDF_observed
        if self.texture_enabled:
            self.color = submap.color
        if self.enable_esdf:
            self.ESDF = submap.ESDF
            self.fixed = submap.fixed
            self.parent_dir = submap.parent_dir
            self.observed = submap.observed
    
    @ti.kernel
    def init_fields(self):
        for i in range(self.max_disp_particles):
            self.export_color[i] = ti.Vector([0.5, 0.5, 0.5], ti.f32)
            self.export_x[i] = ti.Vector([-100000, -100000, -100000], ti.f32)
            self.export_TSDF_xyz[i] = ti.Vector([-100000, -100000, -100000], ti.f32)

    @ti.kernel
    def init_sphere(self):
        voxels = 30
        radius = self.voxel_size*3
        for i in range(self.N/2-voxels/2, self.N/2+voxels/2):
            for j in range(self.N/2-voxels/2, self.N/2+voxels/2):
                for k in range(self.Nz/2-voxels/2, self.Nz/2+voxels/2):
                    p = self.ijk_to_xyz([i, j, k])
                    self.TSDF[i, j, k] = p.norm() - radius
                    self.TSDF_observed[i, j, k] = 1
                    self.color[i, j, k] = self.colormap[int((p[2]-0.5)/radius*0.5*1024)]
    @ti.func
    def constrain_coor(self, _i):
        _i = _i.cast(ti.i32)
        for d in ti.static(range(3)):
            if _i[d] >= self.N_[d]:
                _i[d] = self.N_[d] - 1
            if _i[d] < 0:
                _i[d] = 0
        return _i

    @ti.func
    def assert_coor(self, _i):
        for d in ti.static(range(3)):
            assert self.N_[d] > _i[d] >= 0 

    @ti.func
    def ijk_to_xyz(self, ijk):
        return (ijk - self.NC_)*self.voxel_size_

    @ti.func
    def i_j_k_to_xyz(self, i, j, k):
        return self.ijk_to_xyz(ti.Vector([i, j, k], ti.f32))

    @ti.func
    def xyz_to_ijk(self, xyz):
        ijk =  xyz / self.voxel_size_ + self.NC_
        return self.constrain_coor(ijk)

    @ti.func 
    def w_x_p(self, d, z):
        epi = ti.static(self.voxel_size)
        theta = ti.static(self.voxel_size*4)
        ret = 0.0
        if d > ti.static(-epi):
            ret = 1.0/(z*z)
        elif d > ti.static(-theta):
            ret = (d+theta)/(z*z*(theta-epi))
        return ret
    
    @ti.func
    def is_fixed(self, ijk):
        return ti.abs(self.TSDF[ijk]) < ti.static(self.gamma)
    
    @ti.func
    def insert_lower(self, ijk):
        if self.num_lower_queue[None] >= self.max_queue_size:
            print("lower queue exceeds")
            assert self.num_lower_queue[None] < self.max_queue_size

        elif ti.is_active(self.Broot, ijk):
            #Only add activated voxel
            self.assert_coor(ijk)
            assert self.num_lower_queue[None] < self.max_queue_size
            _index = ti.atomic_add(self.num_lower_queue[None], 1)
            self.lower_queue[_index] = ijk

    @ti.func
    def insert_raise(self, ijk):
        if ti.is_active(self.Broot, ijk):
            #Only add activated voxel
            self.assert_coor(ijk)
            assert self.num_raise_queue[None] < self.max_queue_size
            _index = ti.atomic_add(self.num_raise_queue[None], 1)
            self.raise_queue[_index] = ijk
                

    @ti.func
    def insertneighbors_lower(self, ijk):
        for dir in ti.static(self.neighbors):
            self.insert_lower(ijk+dir)

    @ti.func
    def process_raise_queue(self):
        print("Processing raise queue")
        while self.head_raise_queue[None] < self.num_raise_queue[None]:
            _index_head = self.raise_queue[ti.atomic_add(self.head_raise_queue[None], 1)]
            self.ESDF[_index_head] = sign(self.ESDF[_index_head]) * ti.static(self.max_ray_length)
            for dir in ti.static(self.neighbors):
                n_voxel_ijk = dir + _index_head
                self.assert_coor(n_voxel_ijk) #L203
                if all(dir == self.parent_dir[n_voxel_ijk]):
                    #This line of code cause issue
                    self.insert_raise(n_voxel_ijk)
                else:
                    self.insert_lower(n_voxel_ijk)

    @ti.func
    def process_lower_queue(self):
        while self.head_lower_queue[None] < self.num_lower_queue[None]:
            _index_head = self.lower_queue[ti.atomic_add(self.head_lower_queue[None], 1)]

            assert ti.is_active(self.Broot, _index_head)

            for dir in ti.static(self.neighbors):
                n_voxel_ijk = dir + _index_head
                self.assert_coor(n_voxel_ijk) #L219
                if ti.is_active(self.Broot, n_voxel_ijk):
                    dis = dir.norm()*self.voxel_size
                    n_esdf = self.ESDF[n_voxel_ijk] 
                    _n_esdf = self.ESDF[_index_head] + dis
                    if n_esdf > 0 and _n_esdf < n_esdf:
                        self.ESDF[n_voxel_ijk] = _n_esdf
                        self.parent_dir[n_voxel_ijk] = -dir
                        # self.insert_lower(n_voxel_ijk)
                    else:
                        _n_esdf = self.ESDF[_index_head] - dis
                        if n_esdf < 0 and _n_esdf > n_esdf:
                            self.ESDF[n_voxel_ijk] = _n_esdf
                            self.parent_dir[n_voxel_ijk] = -dir
                            # self.insert_lower(n_voxel_ijk)
        print("Lower queue final size", self.head_lower_queue[None])


    @ti.func
    def propogate_esdf(self):
        self.num_raise_queue[None] = 0
        self.num_lower_queue[None] = 0
        self.head_raise_queue[None] = 0
        self.head_lower_queue[None] = 0
        count_update_tsdf = 0
        for i, j, k in self.updated_TSDF:
            _voxel_ijk = ti.Vector([i, j, k], ti.f32)
            t_d = self.TSDF[_voxel_ijk]
            count_update_tsdf += 1
            if self.is_fixed(_voxel_ijk):
                if self.ESDF[_voxel_ijk] > t_d or self.observed[_voxel_ijk] == 0:
                    self.observed[_voxel_ijk] = 1
                    self.ESDF[_voxel_ijk] = t_d
                    self.insert_lower(_voxel_ijk)
                else:
                    self.ESDF[_voxel_ijk] = t_d
                    self.insert_raise(_voxel_ijk)
                    self.insert_lower(_voxel_ijk)
            else:
                if self.fixed[_voxel_ijk] > 0:
                    self.observed[_voxel_ijk] = 1
                    self.ESDF[_voxel_ijk] = sign(t_d)*ti.static(self.max_ray_length)
                    self.insert_raise(_voxel_ijk)
                elif self.observed[_voxel_ijk] == 0:
                    self.observed[_voxel_ijk] = 1
                    self.ESDF[_voxel_ijk] = sign(t_d)*ti.static(self.max_ray_length)
                    self.insertneighbors_lower(_voxel_ijk)
        print("update TSDF block", count_update_tsdf, "raise queue", self.num_raise_queue[None], "lower", self.num_lower_queue[None])
        self.process_raise_queue()
        self.process_lower_queue()

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        if self.enable_esdf:
            self.Troot.deactivate_all()
        self.PCLroot.deactivate_all()
        self.set_pose(R, T)
        self.recast_pcl_to_map_kernel(xyz_array, rgb_array, n)

    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
        if self.enable_esdf:
            self.Troot.deactivate_all()
        self.PCLroot.deactivate_all()
        self.set_pose(R, T)
        TSDF, W_TSDF, TSDF_observed, color = self.active_submap.TSDF, self.active_submap.W_TSDF, self.active_submap.TSDF_observed, self.active_submap.color
        self.recast_depth_to_map_kernel(TSDF, W_TSDF, TSDF_observed, color, depthmap, texture, w, h, K, Kcolor)

    @ti.kernel
    def recast_pcl_to_map_kernel(self, xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]], ti.f32)
            pt = self.input_R[None]@pt
            if ti.static(self.texture_enabled):
                self.process_point(pt, rgb_array[index])
            else:
                self.process_point(pt)
        self.process_new_pcl()

    @ti.kernel
    def recast_depth_to_map_kernel(self, TSDF: ti.template(), W_TSDF: ti.template(), TSDF_observed: ti.template(), color: ti.template(), \
                depthmap: ti.ext_arr(), texture: ti.ext_arr(), w: ti.i32, h: ti.i32, K:ti.ext_arr(), Kcolor:ti.ext_arr()):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        for j in range(h):
            for i in range(w):
                if depthmap[j, i] == 0:
                    continue
                if depthmap[j, i] > ti.static(self.max_ray_length*1000) or depthmap[j, i] < ti.static(self.min_ray_length*1000):
                    continue
                
                dep = ti.cast(depthmap[j, i], ti.f32)/1000.0
                                
                pt = ti.Vector([
                    (ti.cast(i, ti.f32)-cx)*dep/fx, 
                    (ti.cast(j, ti.f32)-cy)*dep/fy, 
                    dep], ti.f32)

                pt_map = self.input_R[None]@pt
                
                if ti.static(self.texture_enabled):
                    fx_c = Kcolor[0]
                    fy_c = Kcolor[4]
                    cx_c = Kcolor[2]
                    cy_c = Kcolor[5]
                    color_i = ti.cast((i-cx)/fx*fx_c+cx_c, ti.int32)
                    color_j = ti.cast((j-cy)/fy*fy_c+cy_c, ti.int32)
                    if color_i < 0 or color_i >= 640 or color_j < 0 or color_j >= 480:
                        continue
                    self.process_point(pt_map, dep, [texture[color_j, color_i, 0], texture[color_j, color_i, 1], texture[color_j, color_i, 2]])
                else:
                    self.process_point(pt_map, dep)
        
        self.process_new_pcl(TSDF, W_TSDF, TSDF_observed, color)

        if ti.static(self.enable_esdf):
            print("ESDF")
            self.propogate_esdf()
    
    @ti.func
    def process_point(self, pt, z, rgb=None):
        pti = self.xyz_to_ijk(pt)
        self.new_pcl_count[pti] += 1
        self.new_pcl_sum_pos[pti] += pt
        self.new_pcl_z[pti] += z
        if ti.static(self.texture_enabled):
            self.new_pcl_sum_color[pti] += rgb

        pti = self.xyz_to_ijk(pt + self.input_T[None])

        self.occupy[pti] += 1

    @ti.func
    def process_new_pcl(self, TSDF, W_TSDF, TSDF_observed, color):
        for i, j, k in self.new_pcl_count:
            if self.new_pcl_count[i, j, k] == 0:
                continue
            c = ti.cast(self.new_pcl_count[i, j, k], ti.f32)
            pos_s2p = self.new_pcl_sum_pos[i, j, k]/c
            len_pos_s2p = pos_s2p.norm()
            d_s2p = pos_s2p /len_pos_s2p
            pos_p = pos_s2p + self.input_T[None]
            z = self.new_pcl_z[i, j, k]/c
            cur_color = ti.Vector([0., 0., 0.], ti.f32)
            if ti.static(self.texture_enabled):
                cur_color = self.new_pcl_sum_color[i, j, k]/c/255.0
            j_f = 0.0
            ray_cast_voxels = ti.min(len_pos_s2p/self.voxel_size+ti.static(self.internal_voxels), self.max_ray_length/self.voxel_size)
            for _j in range(ray_cast_voxels):
                j_f += 1.0
                x_ = d_s2p*j_f*self.voxel_size + self.input_T[None]
                indices = self.xyz_to_ijk(x_)

                #v2p: vector from current voxel to point, e.g. p-x
                #pos_s2p sensor to point
                v2p = pos_p - x_
                d_x_p = v2p.norm()
                d_x_p_s = d_x_p*sign(v2p.dot(pos_s2p))

                w_x_p = self.w_x_p(d_x_p, z)

                TSDF[indices] =  (TSDF[indices]*W_TSDF[indices]+w_x_p*d_x_p_s)/(W_TSDF[indices]+w_x_p)
                TSDF_observed[indices] += 1
                
                W_TSDF[indices] = ti.min(W_TSDF[indices]+w_x_p, Wmax)
                if ti.static(self.enable_esdf):
                    self.updated_TSDF[indices] = 1
                if ti.static(self.texture_enabled):
                    color[indices] = cur_color
            self.new_pcl_count[i, j, k] = 0

    @ti.kernel
    def cvt_occupy_to_voxels(self):
        # Number for level
        self.num_export_particles[None] = 0
        for i, j, k in self.occupy:
            if self.occupy[i, j, k] > self.min_occupy_thres:
                index = ti.atomic_add(self.num_export_particles[None], 1)
                if self.num_export_particles[None] < self.max_disp_particles:
                    self.export_x[index] = self.i_j_k_to_xyz(i, j, k)
                    if ti.static(self.texture_enabled):
                        self.export_color[index] = self.color[i, j, k]

    @ti.kernel
    def cvt_TSDF_surface_to_voxels(self):
        # Number for TSDF
        self.num_export_TSDF_particles[None] = 0
        for i, j, k in self.TSDF:
            if self.occupy[i, j, k] and ti.abs(self.TSDF[i, j, k] ) < self.tsdf_surface_thres:
                index = ti.atomic_add(self.num_export_TSDF_particles[None], 1)
                if self.num_export_TSDF_particles[None] < self.max_disp_particles:
                    self.export_TSDF[index] = self.TSDF[i, j, k]
                    if ti.static(self.texture_enabled):
                        self.export_color[index] = self.color[i, j, k]
                    self.export_TSDF_xyz[index] = self.i_j_k_to_xyz(i, j, k)

    @ti.kernel
    def cvt_ESDF_to_voxels_slice(self, z: ti.template()):
        # Number for ESDF
        self.num_export_ESDF_particles[None] = 0
        for i, j, k in self.ESDF:
            _index = (z+self.map_size_[2]/2)/self.voxel_size
            if _index - 0.5 < k < _index + 0.5:
                index = ti.atomic_add(self.num_export_ESDF_particles[None], 1)
                if self.num_export_ESDF_particles[None] < self.max_disp_particles:
                    self.export_ESDF[index] = self.ESDF[i, j, k]
                    self.export_ESDF_xyz[index] = self.i_j_k_to_xyz(i, j, k)
    
    @ti.kernel
    def cvt_TSDF_to_voxels_slice_kernel(self, z: ti.template(), dz:ti.template()):
        _index = int((z+self.map_size_[2]/2.0)/self.voxel_size)
        # Number for ESDF
        self.num_export_TSDF_particles[None] = 0
        for i, j, k in self.TSDF:
            if self.TSDF_observed[i, j, k] > 0:
                if _index - dz < k < _index + dz:
                    index = ti.atomic_add(self.num_export_TSDF_particles[None], 1)
                    if self.num_export_TSDF_particles[None] < self.max_disp_particles:
                        self.export_TSDF[index] = self.TSDF[i, j, k]
                        self.export_TSDF_xyz[index] = self.i_j_k_to_xyz(i, j, k)
                        _c = int(max(min((self.TSDF[i, j, k]*self.color_rate + 0.5)*1024, 1024), 0))
                        self.export_color[index] = self.colormap[_c]

    def cvt_TSDF_to_voxels_slice(self, z, dz=0.5):
        self.cvt_TSDF_to_voxels_slice_kernel(z, dz)

    def get_voxels_occupy(self):
        self.get_occupy_to_voxels()
        return self.export_x.to_numpy(), self.export_color.to_numpy()
    
    def get_voxels_TSDF_surface(self):
        self.cvt_TSDF_surface_to_voxels()
        if self.texture_enabled:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), self.export_color.to_numpy()
        else:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), None
        
    def get_voxels_TSDF_slice(self, z):
        self.cvt_TSDF_to_voxels_slice(z)
        return self.export_ESDF_xyz.to_numpy(), self.export_TSDF.to_numpy(),

    def get_voxels_ESDF_slice(self, z):
        self.cvt_ESDF_to_voxels_slice(z)
        return self.export_ESDF_xyz.to_numpy(), self.export_ESDF.to_numpy(),