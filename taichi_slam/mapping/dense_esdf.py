# This file is an easy voxblox implentation based on taichi lang

import taichi as ti
import math
from .mapping_common import *

Wmax = 1000

var = [1, 2, 3, 4, 5]
@ti.data_oriented
class DenseSDF(BaseMap):
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=0, texture_enabled=False, \
            max_disp_particles=1000000, num_voxel_per_blk_axis=16, max_ray_length=10, min_ray_length=0.3, 
            enable_esdf=False, internal_voxels = 10, max_submap_size=1000, is_global_map=False, 
            disp_ceiling=1.8, disp_floor=-0.3):
        super(DenseSDF, self).__init__()
        self.map_size_xy = map_scale[0]
        self.map_size_z = map_scale[1]

        self.num_voxel_per_blk_axis = num_voxel_per_blk_axis
        self.voxel_size = voxel_size

        self.N = math.ceil(map_scale[0] / voxel_size/num_voxel_per_blk_axis)*num_voxel_per_blk_axis
        self.Nz = math.ceil(map_scale[1] / voxel_size/num_voxel_per_blk_axis)*num_voxel_per_blk_axis

        self.block_num_xy = math.ceil(map_scale[0] / voxel_size/num_voxel_per_blk_axis)
        self.block_num_z = math.ceil(map_scale[1] / voxel_size/num_voxel_per_blk_axis)

        self.map_size_xy = voxel_size * self.N
        self.map_size_z = voxel_size * self.Nz

        self.max_disp_particles = max_disp_particles
        self.min_occupy_thres = min_occupy_thres

        self.enable_texture = texture_enabled

        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length
        self.tsdf_surface_thres = self.voxel_size
        self.gamma = self.voxel_size
        self.enable_esdf = enable_esdf
        self.internal_voxels = internal_voxels
        self.max_submap_size = max_submap_size

        self.clear_last_TSDF_exporting = False
        self.is_global_map = is_global_map
        self.disp_ceiling = disp_ceiling
        self.disp_floor = disp_floor

        self.initialize_fields()

    def data_structures(self, submap_num, block_num_xy, block_num_z, num_voxel_per_blk_axis):
        if num_voxel_per_blk_axis < 1:
            print("num_voxel_per_blk_axis must be greater than 1")
            exit(0)
        if self.is_global_map:
            Broot = ti.root.pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijkl, (1, num_voxel_per_blk_axis, num_voxel_per_blk_axis, num_voxel_per_blk_axis))
        else:
            Broot = ti.root.pointer(ti.i, submap_num)
            B = Broot.pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z)).dense(ti.ijkl, (1, num_voxel_per_blk_axis, num_voxel_per_blk_axis, num_voxel_per_blk_axis))
        return B, Broot
    
    def data_structures_grouped(self, block_num_xy, block_num_z, num_voxel_per_blk_axis):
        if num_voxel_per_blk_axis > 1:
            Broot = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijk, (num_voxel_per_blk_axis, num_voxel_per_blk_axis, num_voxel_per_blk_axis))
        else:
            B = ti.root.dense(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            Broot = B
        return B, Broot

    def initialize_sdf_fields(self):
        block_num_xy = self.block_num_xy
        block_num_z = self.block_num_z
        num_voxel_per_blk_axis = self.num_voxel_per_blk_axis
        submap_num = self.max_submap_size
        if self.is_global_map:
            submap_num = 1

        self.TSDF = ti.field(dtype=ti.f32)
        self.W_TSDF = ti.field(dtype=ti.f32)
        self.TSDF_observed = ti.field(dtype=ti.i32)
        if self.enable_texture:
            self.color = ti.Vector.field(3, dtype=ti.f32)
        else:
            self.color = None
        if self.enable_esdf:
            self.ESDF = ti.field(dtype=ti.f32)
            self.observed = ti.field(dtype=ti.i8)
            self.fixed = ti.field(dtype=ti.i8)
            self.parent_dir = ti.Vector.field(3, dtype=ti.i32)
        self.B, self.Broot = self.data_structures(submap_num, block_num_xy, block_num_z, num_voxel_per_blk_axis)
        self.B.place(self.W_TSDF,self.TSDF, self.TSDF_observed)
        if self.enable_esdf:
            self.B.place(self.ESDF, self.observed, self.fixed, self.parent_dir)
        if self.enable_texture:
            self.B.place(self.color)
        if self.enable_esdf:
            self.updated_TSDF = ti.field(dtype=ti.i32)
            self.T, self.Troot = self.data_structures(submap_num, block_num_xy, block_num_z, num_voxel_per_blk_axis)
            self.T.place(self.updated_TSDF)
        
    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_TSDF_particles = ti.field(dtype=ti.i32, shape=())
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

        self.new_pcl_count = ti.field(dtype=ti.i32)
        self.new_pcl_sum_pos = ti.Vector.field(3, dtype=ti.f32) #position in sensor coor
        self.new_pcl_z = ti.field(dtype=ti.f32) #position in sensor coor
        self.PCL, self.PCLroot = self.data_structures_grouped(self.block_num_xy, self.block_num_z, self.num_voxel_per_blk_axis)
        self.PCL.place(self.new_pcl_count, self.new_pcl_sum_pos, self.new_pcl_z)

        self.initialize_sdf_fields()
        if self.enable_texture:
            self.new_pcl_sum_color = ti.Vector.field(3, dtype=ti.f32)

        self.max_queue_size = 1000000
        self.raise_queue = ti.Vector.field(3, dtype=ti.i32, shape=self.max_queue_size)
        self.lower_queue = ti.Vector.field(3, dtype=ti.i32, shape=self.max_queue_size)
        self.num_raise_queue = ti.field(dtype=ti.i32, shape=())
        self.num_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_lower_queue = ti.field(dtype=ti.i32, shape=())
        self.head_raise_queue = ti.field(dtype=ti.i32, shape=())

        if self.enable_texture:
            self.PCL.place(self.new_pcl_sum_color)

        self.neighbors = []
        for _di in range(-1, 2):
            for _dj in range(-1, 2):
                for _dk in range(-1, 2):
                    if _di !=0 or _dj !=0 or _dk != 0:
                        self.neighbors.append(ti.Vector([_di, _dj, _dk], ti.f32))
        
        self.init_colormap()
        self.init_fields()
        self.initialize_submap_fields(self.max_submap_size)

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
        print(radius)
        for i in range(self.N/2-voxels/2, self.N/2+voxels/2):
            for j in range(self.N/2-voxels/2, self.N/2+voxels/2):
                for k in range(self.Nz/2-voxels/2, self.Nz/2+voxels/2):
                    p = self.ijk_to_xyz([i, j, k])
                    self.TSDF[i, j, k] = p.norm() - radius
                    self.TSDF_observed[i, j, k] = 1
                    self.color[i, j, k] = self.colormap[int((p[2]-0.5)/radius*0.5*1024)]
    @ti.func
    def constrain_coor(self, _i):
        ijk = _i.cast(ti.i32)
        for d in ti.static(range(3)):
            if ijk[d] >= self.N_[d]:
                ijk[d] = self.N_[d] - 1
            if ijk[d] < 0:
                ijk[d] = 0
        return ijk

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
    def submap_i_j_k_to_xyz(self, s, i, j, k):
        ijk = self.ijk_to_xyz(ti.Vector([i, j, k], ti.f32))
        return self.submaps_base_R[s]@ijk + self.submaps_base_T[s]

    @ti.func
    def xyz_to_ijk(self, xyz):
        ijk =  xyz / self.voxel_size_ + self.NC_
        return self.constrain_coor(ijk)

    @ti.func
    def xyz_to_0ijk(self, xyz):
        ijk =  xyz / self.voxel_size_ + self.NC_
        _ijk = self.constrain_coor(ijk)
        return ti.Vector([0, _ijk[0], _ijk[1], _ijk[2]], ti.i32)

    @ti.func
    def sxyz_to_ijk(self, s, xyz):
        ijk =  xyz / self.voxel_size_ + self.NC_
        ijk_ = self.constrain_coor(ijk)
        return [s, ijk_[0], ijk_[1], ijk_[2]]

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
        self.recast_depth_to_map_kernel(depthmap, texture, w, h, K, Kcolor)

    @ti.kernel
    def recast_pcl_to_map_kernel(self, xyz_array: ti.types.ndarray(), rgb_array: ti.types.ndarray(), n: ti.i32):
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]], ti.f32)
            pt = self.input_R[None]@pt
            if ti.static(self.enable_texture):
                self.process_point(pt, rgb_array[index])
            else:
                self.process_point(pt)
        self.process_new_pcl()

    @ti.kernel
    def recast_depth_to_map_kernel(self, depthmap: ti.types.ndarray(), texture: ti.types.ndarray(), w: ti.i32, h: ti.i32, K:ti.types.ndarray(), Kcolor:ti.types.ndarray()):
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
                
                if ti.static(self.enable_texture):
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
        self.process_new_pcl()

        if ti.static(self.enable_esdf):
            print("ESDF")
            self.propogate_esdf()
    
    @ti.func
    def process_point(self, pt, z, rgb=None):
        pti = self.xyz_to_ijk(pt)
        self.new_pcl_count[pti] += 1
        self.new_pcl_sum_pos[pti] += pt
        self.new_pcl_z[pti] += z
        if ti.static(self.enable_texture):
            self.new_pcl_sum_color[pti] += rgb

    @ti.func
    def process_new_pcl(self):
        submap_id = self.active_submap_id[None]
        for i, j, k in self.new_pcl_count:
            if self.new_pcl_count[i, j, k] == 0:
                continue
            c = ti.cast(self.new_pcl_count[i, j, k], ti.f32)
            pos_s2p = self.new_pcl_sum_pos[i, j, k]/c
            len_pos_s2p = pos_s2p.norm()
            d_s2p = pos_s2p /len_pos_s2p
            pos_p = pos_s2p + self.input_T[None]
            z = self.new_pcl_z[i, j, k]/c

            j_f = 0.0
            ray_cast_voxels = ti.min(len_pos_s2p/self.voxel_size+ti.static(self.internal_voxels), self.max_ray_length/self.voxel_size)
            for _j in range(ray_cast_voxels):
                j_f += 1.0
                x_ = d_s2p*j_f*self.voxel_size + self.input_T[None]
                xi = self.sxyz_to_ijk(submap_id, x_)

                #v2p: vector from current voxel to point, e.g. p-x
                #pos_s2p sensor to point
                v2p = pos_p - x_
                d_x_p = v2p.norm()
                d_x_p_s = d_x_p*sign(v2p.dot(pos_s2p))

                w_x_p = self.w_x_p(d_x_p, z)

                self.TSDF[xi] =  (self.TSDF[xi]*self.W_TSDF[xi]+w_x_p*d_x_p_s)/(self.W_TSDF[xi]+w_x_p)
                self.TSDF_observed[xi] += 1
                
                self.W_TSDF[xi] = ti.min(self.W_TSDF[xi]+w_x_p, Wmax)
                if ti.static(self.enable_esdf):
                    self.updated_TSDF[xi] = 1
                if ti.static(self.enable_texture):
                    self.color[xi] = self.new_pcl_sum_color[i, j, k]/c/255.0
            self.new_pcl_count[i, j, k] = 0

    def cvt_occupy_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels()

    def cvt_TSDF_surface_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels_kernel(self.num_TSDF_particles, 
                self.export_TSDF_xyz, self.export_color, self.max_disp_particles,
                self.clear_last_TSDF_exporting, False)
        self.clear_last_TSDF_exporting = False

    def cvt_TSDF_surface_to_voxels_to(self, num_TSDF_particles, max_disp_particles, export_TSDF_xyz, export_color):
        self.cvt_TSDF_surface_to_voxels_kernel(num_TSDF_particles, 
                export_TSDF_xyz, export_color, max_disp_particles, False, True)

    @ti.kernel
    def cvt_TSDF_surface_to_voxels_kernel(self, num_TSDF_particles:ti.template(), export_TSDF_xyz:ti.template(),
            export_color:ti.template(), max_disp_particles:ti.template(), clear_last:ti.template(), add_to_cur:ti.template()):
        # Number for TSDF
        if clear_last:
            for i in range(num_TSDF_particles[None]):
                export_color[i] = ti.Vector([0.5, 0.5, 0.5], ti.f32)
                export_TSDF_xyz[i] = ti.Vector([-100000, -100000, -100000], ti.f32)
        
        if not add_to_cur:
            num_TSDF_particles[None] = 0
        
        disp_floor, disp_ceiling = ti.static(self.disp_floor, self.disp_ceiling)

        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k]:
                    if ti.abs(self.TSDF[s, i, j, k] ) < self.tsdf_surface_thres*2:
                        xyz = ti.Vector([0., 0., 0.], ti.f32)
                        if ti.static(self.is_global_map):
                            xyz = self.i_j_k_to_xyz(i, j, k)
                        else:
                            xyz = self.submap_i_j_k_to_xyz(s, i, j, k)
                        if xyz[2] > disp_ceiling or xyz[2] < disp_floor:
                            continue
                        index = ti.atomic_add(num_TSDF_particles[None], 1)
                        if num_TSDF_particles[None] < max_disp_particles:
                            if ti.static(self.enable_texture):
                                export_color[index] = self.color[s, i, j, k]
                                export_TSDF_xyz[index] = xyz
                            else:
                                export_color[index] = self.color_from_colomap(xyz[2], disp_floor, disp_ceiling)
                                export_TSDF_xyz[index] = xyz

    @ti.kernel
    def cvt_ESDF_to_voxels_slice(self, z: ti.template()):
        # Number for ESDF
        self.num_export_ESDF_particles[None] = 0
        for s, i, j, k in self.ESDF:
            if s == self.active_submap_id[None]:
                _index = (z+self.map_size_[2]/2)/self.voxel_size
                if _index - 0.5 < k < _index + 0.5:
                    index = ti.atomic_add(self.num_export_ESDF_particles[None], 1)
                    if self.num_export_ESDF_particles[None] < self.max_disp_particles:
                        self.export_ESDF[index] = self.ESDF[i, j, k]
                        self.export_ESDF_xyz[index] = self.submap_i_j_k_to_xyz(s, i, j, k)
    
    @ti.kernel
    def cvt_TSDF_to_voxels_slice_kernel(self, z: ti.template(), dz:ti.template()):
        _index = int((z+self.map_size_[2]/2.0)/self.voxel_size)
        # Number for ESDF
        self.num_TSDF_particles[None] = 0
        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k] > 0:
                    if _index - dz < k < _index + dz:
                        index = ti.atomic_add(self.num_TSDF_particles[None], 1)
                        if self.num_TSDF_particles[None] < self.max_disp_particles:
                            self.export_TSDF[index] = self.TSDF[s, i, j, k]
                            if ti.static(self.is_global_map):
                                self.export_TSDF_xyz[index] = self.i_j_k_to_xyz(i, j, k)
                            else:
                                self.export_TSDF_xyz[index] = self.submap_i_j_k_to_xyz(s, i, j, k)
                            self.export_color[index] = self.color_from_colomap(self.TSDF[s, i, j, k], -0.5, 0.5)

    @ti.kernel
    def fuse_submaps_kernel(self, TSDF:ti.template(), W_TSDF:ti.template(), TSDF_observed:ti.template(), color:ti.template()):
        for s, i, j, k in TSDF:
            if TSDF_observed[s, i, j, k] > 0:
                tsdf = TSDF[s, i, j, k]
                w_tsdf = W_TSDF[s, i, j, k]
                xyz = self.submap_i_j_k_to_xyz(s, i, j, k)
                ijk = self.xyz_to_0ijk(xyz)
                #Naive merging with weight. TODO: use interpolation
                w_new = w_tsdf + self.W_TSDF[ijk]
                self.TSDF[ijk] = (self.W_TSDF[ijk]*self.TSDF[ijk] + w_tsdf*tsdf)/w_new
                if ti.static(self.enable_texture):
                    c = color[s, i, j, k]
                    self.color[ijk] = (self.W_TSDF[ijk]*self.color[ijk] + w_tsdf*c)/w_new
                self.W_TSDF[ijk] = w_new
                self.TSDF_observed[ijk] = 1

    def fuse_submaps(self, submaps):
        self.B.parent().deactivate_all()
        print("try to fuse all submaps, currently active submap", submaps.active_submap_id[None])
        self.fuse_submaps_kernel(submaps.TSDF, submaps.W_TSDF, submaps.TSDF_observed, submaps.color)

    def cvt_TSDF_to_voxels_slice(self, z, dz=0.5):
        self.cvt_TSDF_to_voxels_slice_kernel(z, dz)

    def get_voxels_occupy(self):
        self.get_occupy_to_voxels()
        return self.export_x.to_numpy(), self.export_color.to_numpy()
    
    def get_voxels_TSDF_surface(self):
        self.cvt_TSDF_surface_to_voxels()
        if self.enable_texture:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), self.export_color.to_numpy()
        else:
            return self.export_TSDF_xyz.to_numpy(), self.export_TSDF.to_numpy(), None
        
    def get_voxels_TSDF_slice(self, z):
        self.cvt_TSDF_to_voxels_slice(z)
        return self.export_ESDF_xyz.to_numpy(), self.export_TSDF.to_numpy()

    def get_voxels_ESDF_slice(self, z):
        self.cvt_ESDF_to_voxels_slice(z)
        return self.export_ESDF_xyz.to_numpy(), self.export_ESDF.to_numpy()
    