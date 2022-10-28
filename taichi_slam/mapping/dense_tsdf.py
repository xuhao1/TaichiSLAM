# This file is an easy voxblox implentation based on taichi lang

import taichi as ti
import math
from .mapping_common import *
import time

Wmax = 1000

var = [1, 2, 3, 4, 5]
@ti.data_oriented
class DenseTSDF(BaseMap):
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, texture_enabled=False, \
            max_disp_particles=1000000, num_voxel_per_blk_axis=16, max_ray_length=10, min_ray_length=0.3,
            internal_voxels = 10, max_submap_size=1000, is_global_map=False, 
            disp_ceiling=1.8, disp_floor=-0.3):
        super(DenseTSDF, self).__init__(voxel_size)
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

        self.enable_texture = texture_enabled

        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length
        self.tsdf_surface_thres = self.voxel_size*1.5
        self.internal_voxels = internal_voxels
        self.max_submap_size = max_submap_size

        self.is_global_map = is_global_map
        self.disp_ceiling = disp_ceiling
        self.disp_floor = disp_floor

        self.initialize_fields()
        print(f"TSDF map initialized blocks {self.block_num_xy}x{self.block_num_xy}x{self.block_num_z}")

    def data_structures(self, submap_num, block_num_xy, block_num_z, num_voxel_per_blk_axis):
        if num_voxel_per_blk_axis < 1:
            print("num_voxel_per_blk_axis must be greater than 1")
            exit(0)
        if self.is_global_map:
            Broot = ti.root.pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijkl, (1, num_voxel_per_blk_axis, num_voxel_per_blk_axis, num_voxel_per_blk_axis))
        else:
            Broot = ti.root.pointer(ti.i, submap_num).pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijkl, (1, num_voxel_per_blk_axis, num_voxel_per_blk_axis, num_voxel_per_blk_axis))
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
        
        offset = [0, -self.N//2, -self.N//2, -self.Nz//2]

        self.TSDF = ti.field(dtype=ti.f16)
        self.W_TSDF = ti.field(dtype=ti.f16)
        self.TSDF_observed = ti.field(dtype=ti.i8)
        self.occupy = ti.field(dtype=ti.i8)
        if self.enable_texture:
            self.color = ti.Vector.field(3, dtype=ti.f16)
        else:
            self.color = None
        self.B, self.Broot = self.data_structures(submap_num, block_num_xy, block_num_z, num_voxel_per_blk_axis)
        self.B.place(self.W_TSDF,self.TSDF, self.TSDF_observed, self.occupy, offset=offset)
        if self.enable_texture:
            self.B.place(self.color, offset=offset)
        self.mem_per_voxel = 2 + 2 + 1 + 1
        if self.enable_texture:
            self.mem_per_voxel += 6
        
    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_TSDF_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_ESDF_particles = ti.field(dtype=ti.i32, shape=())

        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        
        self.NC_ = ti.Vector([self.N//2, self.N//2, self.Nz//2], ti.i32)

        self.new_pcl_count = ti.field(dtype=ti.i32)
        self.new_pcl_sum_pos = ti.Vector.field(3, dtype=ti.f16) #position in sensor coor
        self.new_pcl_z = ti.field(dtype=ti.f16) #position in sensor coor
        grp_block_num = max(int(3.2*self.max_ray_length/self.num_voxel_per_blk_axis/self.voxel_size), 1)
        self.PCL, self.PCLroot = self.data_structures_grouped(grp_block_num, grp_block_num, self.num_voxel_per_blk_axis)
        offset = [-self.num_voxel_per_blk_axis*grp_block_num//2, -self.num_voxel_per_blk_axis*grp_block_num//2, -self.num_voxel_per_blk_axis*grp_block_num//2]
        self.PCL.place(self.new_pcl_count, self.new_pcl_sum_pos, self.new_pcl_z, offset=offset)

        self.slice_z = ti.field(dtype=ti.f16, shape=())

        self.initialize_sdf_fields()
        if self.enable_texture:
            self.new_pcl_sum_color = ti.Vector.field(3, dtype=ti.f16)
            self.PCL.place(self.new_pcl_sum_color, offset=offset)

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
        for i in range(self.N/2-voxels/2, self.N/2+voxels/2):
            for j in range(self.N/2-voxels/2, self.N/2+voxels/2):
                for k in range(self.Nz/2-voxels/2, self.Nz/2+voxels/2):
                    p = self.ijk_to_xyz([i, j, k])
                    self.TSDF[i, j, k] = p.norm() - radius
                    self.TSDF_observed[i, j, k] = 1
                    self.color[i, j, k] = self.colormap[int((p[2]-0.5)/radius*0.5*1024)]

    @ti.func
    def is_unobserved(self, sijk):
        return self.TSDF_observed[sijk] == 0

    @ti.func
    def is_occupy(self, sijk):
        occ1 = self.occupy[sijk] > 0
        occ2 = self.TSDF[sijk] < self.tsdf_surface_thres
        return occ1 or occ2

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array):
        self.PCLroot.deactivate_all()
        self.set_pose(R, T)
        self.recast_pcl_to_map_kernel(xyz_array, rgb_array)

    def recast_depth_to_map(self, R, T, depthmap, texture):
        self.PCLroot.deactivate_all()
        self.set_pose(R, T)
        self.recast_depth_to_map_kernel(depthmap, texture)

    @ti.kernel
    def recast_pcl_to_map_kernel(self, xyz_array: ti.types.ndarray(), rgb_array: ti.types.ndarray()):
        n = xyz_array.shape[0]
        for index in range(n):
            pt = ti.Vector([
                xyz_array[index,0], 
                xyz_array[index,1], 
                xyz_array[index,2]], ti.f32)
            pt = self.input_R[None]@pt
            pt_length = pt.norm()
            if pt_length < ti.static(self.max_ray_length):
                if ti.static(self.enable_texture):
                    rgb = ti.Vector([
                        rgb_array[index,0], 
                        rgb_array[index,1], 
                        rgb_array[index,2]], ti.f16)
                    self.process_point(pt, pt.norm(), rgb)
                else:
                    self.process_point(pt, pt.norm())
        self.process_new_pcl()

    @ti.kernel
    def recast_depth_to_map_kernel(self, depthmap: ti.types.ndarray(), texture: ti.types.ndarray()):
        h = depthmap.shape[0]
        w = depthmap.shape[1]
        for j in range(h):
            for i in range(w):
                if depthmap[j, i] == 0:
                    continue
                if depthmap[j, i] > ti.static(self.max_ray_length*1000) or depthmap[j, i] < ti.static(self.min_ray_length*1000):
                    continue
                
                dep = ti.cast(depthmap[j, i], ti.f32)/1000.0
                pt = self.unproject_point_dep(i, j, dep)
                pt_map = self.input_R[None]@pt
                if ti.static(self.enable_texture):
                    colorij = self.color_ind_from_depth_pt(i, j, texture.shape[1], texture.shape[0])
                    color = [texture[colorij, 0], texture[colorij, 1], texture[colorij, 2]]
                    self.process_point(pt_map, dep, color)
                else:
                    self.process_point(pt_map, dep)
        self.process_new_pcl()
        
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
            c = ti.cast(self.new_pcl_count[i, j, k], ti.f16)
            pos_s2p = self.new_pcl_sum_pos[i, j, k]/c
            len_pos_s2p = pos_s2p.norm()
            d_s2p = pos_s2p /len_pos_s2p
            pos_p = pos_s2p + self.input_T[None]
            z = self.new_pcl_z[i, j, k]/c
            j_f = 0.0
            self.occupy[self.sxyz_to_ijk(submap_id, pos_p)] = 1
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
                self.TSDF_observed[xi] = 1
                
                self.W_TSDF[xi] = ti.min(self.W_TSDF[xi]+w_x_p, Wmax)
                if ti.static(self.enable_texture):
                    self.color[xi] = self.new_pcl_sum_color[i, j, k]/c/255.0
            self.new_pcl_count[i, j, k] = 0

    @ti.kernel
    def fuse_submaps_kernel(self, num_submaps: ti.i32, TSDF:ti.template(), W_TSDF:ti.template(), 
            TSDF_observed:ti.template(), occ:ti.template(), color:ti.template(),
            submaps_base_R_np: ti.types.ndarray(), submaps_base_T_np: ti.types.ndarray()):
        for s in range(num_submaps):
            for i in range(3):
                self.submaps_base_T[s][i] = submaps_base_T_np[s, i]
                for j in range(3):
                    self.submaps_base_R[s][i, j] = submaps_base_R_np[s, i, j]

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
                self.occupy[ijk] = self.occupy[ijk] + occ[ijk]

    def fuse_submaps(self, submaps):
        self.B.parent().deactivate_all()
        print("try to fuse all submaps, currently active submap", submaps.active_submap_id[None])
        self.fuse_submaps_kernel(submaps.active_submap_id[None], submaps.TSDF, submaps.W_TSDF, submaps.TSDF_observed, submaps.occupy, 
            submaps.color, self.submaps_base_R_np, self.submaps_base_T_np)

    def cvt_occupy_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels()

    def cvt_TSDF_surface_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels_kernel(self.num_TSDF_particles, 
                self.export_TSDF_xyz, self.export_color, self.max_disp_particles, False)

    def cvt_TSDF_surface_to_voxels_to(self, num_TSDF_particles, max_disp_particles, export_TSDF_xyz, export_color):
        self.cvt_TSDF_surface_to_voxels_kernel(num_TSDF_particles, 
                export_TSDF_xyz, export_color, max_disp_particles, True)
    
    @staticmethod
    @ti.func
    def clear_last_output(num, export_TSDF_xyz, export_color):
        for i in range(num[None]):
            export_color[i] = ti.Vector([0.5, 0.5, 0.5], ti.f32)
            export_TSDF_xyz[i] = ti.Vector([-100000, -100000, -100000], ti.f32)
        num[None] = 0
            
    @ti.kernel
    def cvt_TSDF_surface_to_voxels_kernel(self, num_TSDF_particles:ti.template(), export_TSDF_xyz:ti.template(),
            export_color:ti.template(), max_disp_particles:ti.template(), add_to_cur:ti.template()):
        if not add_to_cur:
            num_TSDF_particles[None] = 0
        
        disp_floor, disp_ceiling = ti.static(self.disp_floor, self.disp_ceiling)

        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k] == 1:
                    if ti.abs(self.TSDF[s, i, j, k] ) < self.tsdf_surface_thres:
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
    def cvt_TSDF_to_voxels_slice_kernel(self, dz:ti.template(), clear_last:ti.template()):
        z = self.slice_z[None]
        dz = ti.static(dz)
        _index = int(z/self.voxel_size)
        # Number for ESDF
        if clear_last:
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

    def cvt_TSDF_to_voxels_slice(self, z, dz=0.5, clear_last=True):
        self.slice_z[None] = z
        self.cvt_TSDF_to_voxels_slice_kernel(dz, clear_last)

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

    @ti.kernel
    def finalization_current_submap(self):
        count = self.count_active_func()/1024
        count_mem = count * ti.static(self.mem_per_voxel)/1024
        # print(f"Will finalize submap {self.active_submap_id[None]} opened voxel: {count}k,{count_mem}MB")

    @ti.func
    def count_active_func(self):
        count = 0
        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k] > 0:
                    ti.atomic_add(count, 1)
        return count

    @ti.kernel
    def count_active(self) -> ti.i32:
        return self.count_active_func()

    @ti.kernel
    def to_numpy(self, data_indices: ti.types.ndarray(element_dim=1), data_tsdf: ti.types.ndarray(), data_wtsdf: ti.types.ndarray(), data_occ: ti.types.ndarray(), data_color:ti.types.ndarray()):
        # Never use it for submap collection! will be extreme large
        count = 0
        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k] > 0:
                    _count = ti.atomic_add(count, 1)
                    data_indices[_count] = [i, j, k]
                    data_tsdf[_count] = self.TSDF[s, i, j, k]
                    data_wtsdf[_count] = self.W_TSDF[s, i, j, k]
                    data_occ[_count] = self.occupy[s, i, j, k]
                    if ti.static(self.enable_texture):
                        data_color[_count] = self.color[s, i, j, k]

    @ti.kernel
    def load_numpy(self, data_indices: ti.types.ndarray(element_dim=1), data_tsdf: ti.types.ndarray(), data_wtsdf: ti.types.ndarray(), data_occ: ti.types.ndarray(), data_color:ti.types.ndarray()):
        for i in range(data_tsdf.shape[0]):
            ind = data_indices[i]
            sijk = 0, ind[0], ind[1], ind[2]
            self.TSDF[sijk] = data_tsdf[i]
            self.W_TSDF[sijk] = data_wtsdf[i]
            self.occupy[sijk] = data_occ[i]
            if ti.static(self.enable_texture):
                self.color[sijk] = data_color[i]
            self.TSDF_observed[sijk] = 1
    
    def export_submap(self):
        s = time.time()
        num = self.count_active()
        indices = np.zeros((num, 3), np.int16)
        tsdf = np.zeros((num), np.float16)
        w_tsdf = np.zeros((num), np.float16)
        occupy = np.zeros((num), np.int8)
        if self.enable_texture:
            color = np.zeros((num, 3), np.float16)
        else:
            color = np.array([])
        self.to_numpy(indices, tsdf, w_tsdf, occupy, color)
        obj = {
            'indices': indices,
            'TSDF': tsdf,
            'W_TSDF': w_tsdf,
            'color': color,
            'occupy': occupy,
            "map_scale": [self.map_size_xy, self.map_size_z],
            "voxel_size": self.voxel_size,
            "texture_enabled": self.enable_texture,
            "num_voxel_per_blk_axis": self.num_voxel_per_blk_axis,
        }
        print(f"Export submap {self.active_submap_id[None]} to numpy, voxels {num/1024:.1f}k, time: {1000*(time.time()-s):.1f}ms")
        return obj
    
    def saveMap(self, filename):
        s = time.time()
        num = self.count_active()
        indices = np.zeros((num, 3), np.int32)
        TSDF = np.zeros((num), np.float32)
        W_TSDF = np.zeros((num), np.float32)
        occupy = np.zeros((num), np.int32)
        if self.enable_texture:
            color = np.zeros((num, 3), np.float32)
        else:
            color = np.array([])
        self.to_numpy(indices, TSDF, W_TSDF, occupy, color)
        obj = {
            'indices': indices,
            'TSDF': TSDF,
            'W_TSDF': W_TSDF,
            'color': color,
            'occupy': occupy,
            "map_scale": [self.map_size_xy, self.map_size_z],
            "voxel_size": self.voxel_size,
            "texture_enabled": self.enable_texture,
            "num_voxel_per_blk_axis": self.num_voxel_per_blk_axis,
        }
        e = time.time()
        print(f"[SubmapMapping] Saving map to {filename} {num} voxels takes {e-s:.1f} seconds")
        np.save(filename, obj)
    
    @staticmethod
    def loadMap(filename):
        obj = np.load(filename, allow_pickle=True).item()
        TSDF = obj['TSDF']
        W_TSDF = obj['W_TSDF']
        color = obj['color']
        indices = obj['indices']
        occupy = obj['occupy']
        mapping = DenseTSDF(map_scale=obj['map_scale'], voxel_size=obj['voxel_size'], 
            texture_enabled=obj['texture_enabled'], num_voxel_per_blk_axis=obj['num_voxel_per_blk_axis'], is_global_map=True)
        mapping.load_numpy(indices, TSDF, W_TSDF, occupy, color)
        print(f"[SubmapMapping] Loaded {TSDF.shape[0]} voxels from {filename}")
        return mapping
