# This file is an easy voxblox implentation based on taichi lang

import taichi as ti
import math
from .mapping_common import *

Wmax = 1000

var = [1, 2, 3, 4, 5]
@ti.data_oriented
class DenseTSDF(Basemap):
    def __init__(self, map_scale=[10, 10], voxel_size=0.05, min_occupy_thres=0, texture_enabled=False, \
            max_disp_particles=1000000, block_size=16, max_ray_length=10, min_ray_length=0.3,
            internal_voxels = 10, max_submap_size=1000, is_global_map=False, 
            disp_ceiling=1.8, disp_floor=-0.3):
        super(DenseTSDF, self).__init__(voxel_size, voxel_size)
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

        self.enable_texture = texture_enabled

        self.max_ray_length = max_ray_length
        self.min_ray_length = min_ray_length
        self.tsdf_surface_thres = self.voxel_size
        self.internal_voxels = internal_voxels
        self.max_submap_size = max_submap_size

        self.require_clear_exported = False
        self.is_global_map = is_global_map
        self.disp_ceiling = disp_ceiling
        self.disp_floor = disp_floor

        self.initialize_fields()

    def data_structures(self, submap_num, block_num_xy, block_num_z, block_size):
        if block_size < 1:
            print("block_size must be greater than 1")
            exit(0)
        if self.is_global_map:
            Broot = ti.root.pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijkl, (1, block_size, block_size, block_size))
        else:
            Broot = ti.root.pointer(ti.i, submap_num)
            B = Broot.pointer(ti.ijkl, (1, block_num_xy, block_num_xy, block_num_z)).dense(ti.ijkl, (1, block_size, block_size, block_size))
        return B, Broot
    
    def data_structures_grouped(self, block_num_xy, block_num_z, block_size):
        if block_size > 1:
            Broot = ti.root.pointer(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            B = Broot.dense(ti.ijk, (block_size, block_size, block_size))
        else:
            B = ti.root.dense(ti.ijk, (block_num_xy, block_num_xy, block_num_z))
            Broot = B
        return B, Broot

    def initialize_sdf_fields(self):
        block_num_xy = self.block_num_xy
        block_num_z = self.block_num_z
        block_size = self.block_size
        submap_num = self.max_submap_size
        if self.is_global_map:
            submap_num = 1
        
        offset = [0, -self.N//2, -self.N//2, -self.Nz//2]

        self.TSDF = ti.field(dtype=ti.f32)
        self.W_TSDF = ti.field(dtype=ti.f32)
        self.TSDF_observed = ti.field(dtype=ti.i32)
        if self.enable_texture:
            self.color = ti.Vector.field(3, dtype=ti.f32)
        else:
            self.color = None
        self.B, self.Broot = self.data_structures(submap_num, block_num_xy, block_num_z, block_size)
        self.B.place(self.W_TSDF,self.TSDF, self.TSDF_observed, offset=offset)
        if self.enable_texture:
            self.B.place(self.color, offset=offset)
        
    def initialize_fields(self):
        self.num_export_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_TSDF_particles = ti.field(dtype=ti.i32, shape=())
        self.num_export_ESDF_particles = ti.field(dtype=ti.i32, shape=())

        self.export_x = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_color = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF = ti.field(dtype=ti.f32, shape=self.max_disp_particles)
        self.export_TSDF_xyz = ti.Vector.field(3, dtype=ti.f32, shape=self.max_disp_particles)
        
        self.NC_ = ti.Vector([self.N/2, self.N/2, self.Nz/2], ti.i32)

        self.new_pcl_count = ti.field(dtype=ti.i32)
        self.new_pcl_sum_pos = ti.Vector.field(3, dtype=ti.f32) #position in sensor coor
        self.new_pcl_z = ti.field(dtype=ti.f32) #position in sensor coor
        self.PCL, self.PCLroot = self.data_structures_grouped(self.block_num_xy, self.block_num_z, self.block_size)
        self.PCL.place(self.new_pcl_count, self.new_pcl_sum_pos, self.new_pcl_z)

        self.initialize_sdf_fields()
        if self.enable_texture:
            self.new_pcl_sum_color = ti.Vector.field(3, dtype=ti.f32)
            self.PCL.place(self.new_pcl_sum_color)

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
    def w_x_p(self, d, z):
        epi = ti.static(self.voxel_size)
        theta = ti.static(self.voxel_size*4)
        ret = 0.0
        if d > ti.static(-epi):
            ret = 1.0/(z*z)
        elif d > ti.static(-theta):
            ret = (d+theta)/(z*z*(theta-epi))
        return ret
    
    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        self.PCLroot.deactivate_all()
        self.set_pose(R, T)
        self.recast_pcl_to_map_kernel(xyz_array, rgb_array, n)

    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
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
                if ti.static(self.enable_texture):
                    self.color[xi] = self.new_pcl_sum_color[i, j, k]/c/255.0
            self.new_pcl_count[i, j, k] = 0

    def cvt_occupy_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels()

    def cvt_TSDF_surface_to_voxels(self):
        self.cvt_TSDF_surface_to_voxels_kernel(self.num_export_TSDF_particles, 
                self.export_TSDF_xyz, self.export_color, self.max_disp_particles,
                self.require_clear_exported , False)
        self.require_clear_exported  = False

    def cvt_TSDF_surface_to_voxels_to(self, num_export_TSDF_particles, max_disp_particles, export_TSDF_xyz, export_color):
        self.cvt_TSDF_surface_to_voxels_kernel(num_export_TSDF_particles, 
                export_TSDF_xyz, export_color, max_disp_particles, False, True)

    @ti.kernel
    def cvt_TSDF_surface_to_voxels_kernel(self, num_export_TSDF_particles:ti.template(), export_TSDF_xyz:ti.template(),
            export_color:ti.template(), max_disp_particles:ti.template(), clear_last:ti.template(), add_to_cur:ti.template()):
        # Number for TSDF
        if clear_last:
            for i in range(num_export_TSDF_particles[None]):
                export_color[i] = ti.Vector([0.5, 0.5, 0.5], ti.f32)
                export_TSDF_xyz[i] = ti.Vector([-100000, -100000, -100000], ti.f32)
        
        if not add_to_cur:
            num_export_TSDF_particles[None] = 0
        
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
                        index = ti.atomic_add(num_export_TSDF_particles[None], 1)
                        if num_export_TSDF_particles[None] < max_disp_particles:
                            if ti.static(self.enable_texture):
                                export_color[index] = self.color[s, i, j, k]
                                export_TSDF_xyz[index] = xyz
                            else:
                                export_color[index] = self.color_from_colomap(xyz[2], disp_floor, disp_ceiling)
                                export_TSDF_xyz[index] = xyz

    @ti.kernel
    def cvt_TSDF_to_voxels_slice_kernel(self, z: ti.template(), dz:ti.template()):
        _index = int(z/self.voxel_size)
        # Number for ESDF
        self.num_export_TSDF_particles[None] = 0
        for s, i, j, k in self.TSDF:
            if s == self.active_submap_id[None]:
                if self.TSDF_observed[s, i, j, k] > 0:
                    if _index - dz < k < _index + dz:
                        index = ti.atomic_add(self.num_export_TSDF_particles[None], 1)
                        if self.num_export_TSDF_particles[None] < self.max_disp_particles:
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