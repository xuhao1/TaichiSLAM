import taichi as ti
import numpy as np
from matplotlib import cm

@ti.func
def sign(val):
    return (0 < val) - (val < 0)

@ti.data_oriented
class BaseMap:
    def __init__(self, voxel_scale):
        self.input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.base_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.base_T = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.base_T_np = np.zeros(3)
        self.base_R_np = np.eye(3)
        self.initialize_base_fields()
        self.frame_id = 0
        self.submap_enabled = False
        self.init_colormap()
        self.voxel_scale = voxel_scale
        self.voxel_scale_ = ti.Vector([voxel_scale, voxel_scale, voxel_scale], ti.f32)
    
    def set_dep_camera_intrinsic(self, K):
        self.K_cam_dep = K
    
    def set_color_camera_intrinsic(self, K):
        self.K_cam_color = K
    
    @ti.func
    def unproject_point_dep(self, i, j, dep):
        fx = ti.static(self.K_cam_dep[0])
        fy = ti.static(self.K_cam_dep[4])
        cx = ti.static(self.K_cam_dep[2])
        cy = ti.static(self.K_cam_dep[5])
        pt = ti.Vector([
            (ti.cast(i, ti.f32)-cx)*dep/fx, 
            (ti.cast(j, ti.f32)-cy)*dep/fy, 
            dep], ti.f32)
        return pt
    
    @ti.func
    def color_ind_from_depth_pt(self, i, j, w, h):
        fx_c = ti.static(self.K_cam_color[0])
        fy_c = ti.static(self.K_cam_color[4])
        cx_c = ti.static(self.K_cam_color[2])
        cy_c = ti.static(self.K_cam_color[5])
        fx = ti.static(self.K_cam_dep[0])
        fy = ti.static(self.K_cam_dep[4])
        cx = ti.static(self.K_cam_dep[2])
        cy = ti.static(self.K_cam_dep[5])

        color_i = ti.cast((i-cx)/fx*fx_c+cx_c, ti.int32)
        color_j = ti.cast((j-cy)/fy*fy_c+cy_c, ti.int32)
        if color_i < 0 or color_i >= h or color_j < 0 or color_j >= w:
            color_i, color_j = 0, 0
        return color_j, color_i

    @ti.kernel
    def initialize_base_fields(self):
        self.input_R[None] = ti.Matrix.identity(ti.f32, 3)
        self.input_T[None] = ti.Matrix.zero(ti.f32, 3)
        self.base_R[None] = ti.Matrix.identity(ti.f32, 3)
        self.base_T[None] = ti.Matrix.zero(ti.f32, 3)

    @ti.kernel
    def random_init_octo(self, pts: ti.template()):
        for i in range(pts):
            x_ = ti.random(dtype = int)%self.N
            y_ = ti.random(dtype = int)%self.N
            z_ = ti.random(dtype = int)%self.Nz
            self.occupy[x_, y_, z_] =  ti.random(dtype = int)%10

    def render_map_to_particles(self, pars, pos_, colors, num_particles_, level):
        pass

    def render_occupy_map_to_particles(self, pars, pos_, colors, num_particles_, voxel_scale):
        if num_particles_ == 0:
            return
        pos = pos_[0:num_particles_,:]
        if not self.enable_texture:
            max_z = np.max(pos[0:num_particles_,2])
            min_z = np.min(pos[0:num_particles_,2])
            colors = cm.jet((pos[0:num_particles_,2] - min_z)/(max_z-min_z))
        pars.set_particles(pos)
        radius = np.ones(num_particles_)*voxel_scale/2
        pars.set_particle_radii(radius)
        pars.set_particle_colors(colors)
    
    def convert_by_base(self, R, T):
        if self.submap_enabled:
            base_R_inv = self.submaps_base_R_np[self.active_submap_id[None]].T
            R_ = base_R_inv @ R
            T_ = base_R_inv @ (T - self.submaps_base_T_np[self.active_submap_id[None]])
        else:
            base_R_inv = self.base_R_np.T
            R_ = base_R_inv @ R
            T_ = base_R_inv @ (T - self.base_T_np)
        return R_, T_
    
    def initialize_submap_fields(self, max_submap_num):
        self.submap_enabled = True
        self.submaps_base_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_submap_num)
        self.submaps_base_T = ti.Vector.field(3, dtype=ti.f32, shape=max_submap_num)
        self.submaps_base_R_np = np.zeros((max_submap_num, 3, 3))
        self.submaps_base_T_np = np.zeros((max_submap_num, 3))
        self.active_submap_id = ti.field(dtype=ti.i32, shape=())
        self.active_submap_id[None] = 0
        self.remote_submap_num = ti.field(dtype=ti.i32, shape=())
        self.remote_submap_num[None] = 0
    
    def get_active_submap_id(self):
        return self.active_submap_id[None]
    
    def switch_to_next_submap(self):
        self.finalization_current_submap()
        self.active_submap_id[None] += 1
        return self.active_submap_id[None]

    def set_base_pose_submap(self, submap_id, _R, _T):
        self.submaps_base_T_np[submap_id] = _T
        self.submaps_base_R_np[submap_id] = _R
        self.set_base_pose_submap_kernel(submap_id, np.ascontiguousarray(_R), np.ascontiguousarray(_T))

    @ti.kernel
    def set_base_pose_submap_kernel(self, submap_id:ti.i16, _R:ti.types.ndarray(), _T:ti.types.ndarray()):
        for i in range(3):
            self.submaps_base_T[submap_id][i] = _T[i]
            for j in range(3):
                self.submaps_base_R[submap_id][i, j] = _R[i, j]

    @ti.func
    def set_base_poses_submap(self, num_submaps, submaps_base_T_np, submaps_base_R_np):
        for s in range(num_submaps):
            for i in range(3):
                self.submaps_base_T[s][i] = submaps_base_T_np[s, i]
                for j in range(3):
                    self.submaps_base_R[s][i, j] = submaps_base_R_np[s, i, j]

    def set_base_pose(self, _R, _T):
        self.base_T_np = _T
        self.base_R_np = _R
        for i in range(3):
            self.base_T[None][i] = _T[i]
            for j in range(3):
                self.base_R[None][i, j] = _R[i, j]

    def set_pose(self, _R, _T):
        _R, _T = self.convert_by_base(_R, _T)
        _R = _R.astype(np.float32)
        _T = _T.astype(np.float32)
        for i in range(3):
            self.input_T[None][i] = _T[i]
            for j in range(3):
                self.input_R[None][i, j] = _R[i, j]

    def init_colormap(self):
        self.colormap = ti.Vector.field(3, float, shape=1024)
        for i in range(1024):
            self.colormap[i][0] = cm.jet(i/1024.0)[0]
            self.colormap[i][1] = cm.jet(i/1024.0)[1]
            self.colormap[i][2] = cm.jet(i/1024.0)[2]
    
    @ti.func
    def raycast(self, pos, dir, max_dist):
        ray_cast_voxels = max_dist/self.voxel_scale_[0]
        x_ = ti.Vector([0., 0., 0.], ti.f32)
        succ = False
        _len = 0.0
        ti.loop_config(serialize=True, parallelize=False)
        for _j in range(ray_cast_voxels):
            _len = _j*self.voxel_scale
            x_ = dir*_len + pos
            if self.is_pos_occupy(x_):
                succ = True
                break
        return succ, x_, _len


    @ti.func
    def is_pos_unobserved(self, xyz):
        submap_id = self.active_submap_id[None]
        ijk = self.sxyz_to_ijk(submap_id, xyz)
        return self.is_unobserved(ijk)

    @ti.func
    def is_pos_occupy(self, xyz):
        submap_id = self.active_submap_id[None]
        ijk = self.sxyz_to_ijk(submap_id, xyz)
        return self.is_occupy(ijk)
    
    @ti.func
    def is_near_pos_occupy(self, xyz, voxel):
        submap_id = self.active_submap_id[None]
        ijk = self.sxyz_to_ijk(submap_id, xyz)
        is_occ = False
        for i in range(-voxel, voxel):
            for j in range(-voxel, voxel):
                for k in range(-voxel, voxel):
                    if self.is_occupy([ijk[0], ijk[1]+i, ijk[2]+j, ijk[3]+k]):
                        is_occ = True
                        break
        return is_occ

    @ti.func
    def is_unobserved(self, ijk):
        print("Not implemented")
        return False

    @ti.func
    def is_occupy(self, ijk):
        print("Not implemented")
        return False
    
    @ti.func 
    def color_from_colomap(self, z, min_z, max_z):
        _c = int(max(min(( (z - min_z)/(max_z-min_z) )*1023, 1023), 0))
        return self.colormap[_c]

    @ti.func
    def ijk_to_xyz(self, ijk):
        return ijk*self.voxel_scale_

    @ti.func
    def i_j_k_to_xyz(self, i, j, k):
        return self.ijk_to_xyz(ti.Vector([i, j, k], ti.f32))

    @ti.func
    def submap_i_j_k_to_xyz(self, s, i, j, k):
        ijk = self.ijk_to_xyz(ti.Vector([i, j, k], ti.f32))
        return self.submaps_base_R[s]@ijk + self.submaps_base_T[s]

    @ti.func
    def sijk_to_xyz(self, sijk):
        s = sijk[0]
        ijk = self.ijk_to_xyz(ti.Vector([sijk[1], sijk[2], sijk[3]], ti.f32))
        return self.submaps_base_R[s]@ijk + self.submaps_base_T[s]
    
    @ti.func
    def xyz_to_ijk(self, xyz):
        ijk =  xyz / self.voxel_scale_
        return self.constrain_coor(ijk)

    @ti.func
    def xyz_to_0ijk(self, xyz):
        ijk =  xyz / self.voxel_scale_
        _ijk = self.constrain_coor(ijk)
        return ti.Vector([0, _ijk[0], _ijk[1], _ijk[2]], ti.i32)

    @ti.func
    def xyz_to_sijk(self, xyz):
        ijk =  xyz / self.voxel_scale_
        _ijk = self.constrain_coor(ijk)
        return ti.Vector([self.active_submap_id[None], _ijk[0], _ijk[1], _ijk[2]], ti.i32)

    @ti.func
    def sxyz_to_ijk(self, s, xyz):
        ijk =  xyz / self.voxel_scale_
        ijk_ = self.constrain_coor(ijk)
        return [s, ijk_[0], ijk_[1], ijk_[2]]

    @ti.func
    def constrain_coor(self, _i):
        ijk = ti.round(_i, ti.i32)
        return ijk
