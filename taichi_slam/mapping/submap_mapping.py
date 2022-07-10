from .dense_sdf import DenseSDF


class SubmapMapping:
    def __init__(self, submap_type=DenseSDF, keyframe_step=20, sub_opts={}, global_opts={}):
        sdf_default_opts = {
            'map_scale': [10, 10],
            'voxel_size': 0.05,
            'min_occupy_thres': 3,
            'texture_enabled': False,
            'min_ray_length': 0.3,
            'max_ray_length': 3.0,
            'max_disp_particles': 100000,
            'block_size': 10,
            'enable_esdf': False
        }
        sdf_default_opts.update(sub_opts)
        self.sub_opts = sdf_default_opts
        self.submaps = {}
        self.frame_count = 0
        self.keyframe_step = keyframe_step
        self.submap_type = submap_type
        self.exporting_global = False
        self.export_TSDF_xyz = None
        self.export_color = None
        self.export_x = None
        self.last_submap = None

        self.global_map = self.create_globalmap(global_opts)

    def create_globalmap(self, global_opts={}):
        sdf_default_opts = {
            'map_scale': [100, 100],
            'voxel_size': 0.05,
            'min_occupy_thres': 3,
            'texture_enabled': False,
            'min_ray_length': 0.3,
            'max_ray_length': 3.0,
            'max_disp_particles': 1000000,
            'block_size': 10,
            'enable_esdf': False
        }
        sdf_default_opts.update(global_opts)
        self.global_map = self.submap_type(**sdf_default_opts)
    
    def set_exporting_global(self):
        self.exporting_global = True
        self.set_export_submap(self.global_map)

    def set_exporting_last_local(self):
        self.exporting_global = False
        if len(self.submaps) > 0:
            self.set_export_submap(self.last_submap)
       
    def set_export_submap(self, new_submap):
        self.export_color = new_submap.export_color
        if self.submap_type == DenseSDF:
            self.export_TSDF_xyz = new_submap.export_TSDF_xyz
            self.num_export_TSDF_particles = new_submap.num_export_TSDF_particles
        else:
            self.export_x = new_submap.export_x

    def create_new_submap(self, frame_id, R, T):
        new_submap = self.submap_type(**self.sub_opts)
        new_submap.set_base_pose(R, T)
        new_submap.frame_id = frame_id
        self.submaps[frame_id] = new_submap

        if not self.exporting_global:
            self.set_export_submap(new_submap)
        self.last_submap = new_submap
        print(f"[SubmapMapping] Created new submap, now have {len(self.submaps)} submaps")
        return new_submap

    def need_create_new_submap(self, R, T):
        if self.frame_count == 0:
            return True
        if self.frame_count % self.keyframe_step == 0:
            return True
        return False

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        pass

    def local_to_global(self):
        pass

    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
        if self.need_create_new_submap(R, T):
            #In early debug we use framecount as frameid
            submap = self.create_new_submap(self.frame_count, R, T)
        else:
            submap = self.last_submap
        submap.recast_depth_to_map(R, T, depthmap, texture, w, h, K, Kcolor)
        self.frame_count += 1
    
    def cvt_TSDF_to_voxels_slice(self, z):
        if len(self.submaps) > 0:
            if self.exporting_global:
                self.global_map.cvt_TSDF_to_voxels_slice(z)
            else:
                self.last_submap.cvt_TSDF_to_voxels_slice(z)

    def cvt_TSDF_surface_to_voxels(self):
        if len(self.submaps) > 0:
            if self.exporting_global:
                self.global_map.cvt_TSDF_surface_to_voxels()
            else:
                self.last_submap.cvt_TSDF_surface_to_voxels()
