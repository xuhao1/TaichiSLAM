from .dense_sdf import DenseSDF

class SubmapMapping:
    def __init__(self, submap_type=DenseSDF, keyframe_step=50, sub_opts={}, global_opts={}):
        sdf_default_opts = {
            'map_scale': [10, 10],
            'voxel_size': 0.05,
            'min_occupy_thres': 3,
            'texture_enabled': False,
            'min_ray_length': 0.3,
            'max_ray_length': 3.0,
            'max_disp_particles': 100000,
            'block_size': 10,
            'enable_esdf': False,
            'max_submap_size': 1000
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
        self.submap_collection = self.submap_type(**self.sub_opts)
        self.global_map = self.create_globalmap(global_opts)
        self.first_init = True
        self.set_exporting_global() # default is exporting local
        # self.set_exporting_local() # default is exporting local

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
            'enable_esdf': False,
            'max_submap_size': 1000,
            'is_global_map': True
        }
        sdf_default_opts.update(global_opts)
        return self.submap_type(**sdf_default_opts)
    
    def set_exporting_global(self):
        self.exporting_global = True
        self.set_export_submap(self.global_map)

    def set_exporting_local(self):
        self.exporting_global = False
        self.set_export_submap(self.submap_collection)
       
    def set_export_submap(self, new_submap):
        self.export_color = new_submap.export_color
        if self.submap_type == DenseSDF:
            self.export_TSDF_xyz = new_submap.export_TSDF_xyz
            self.num_export_TSDF_particles = new_submap.num_export_TSDF_particles
        else:
            self.export_x = new_submap.export_x
    
    def create_new_submap(self, frame_id, R, T):
        if self.first_init:
            self.first_init = False
        else:
            self.submap_collection.switch_to_next_submap()
            self.submap_collection.clear_last_TSDF_exporting = True
            self.local_to_global()
        submap_id = self.submap_collection.get_active_submap_id()
        self.submap_collection.set_base_pose_submap(submap_id, R, T)
        self.global_map.set_base_pose_submap(submap_id, R, T)
        self.submaps[frame_id] = submap_id

        print(f"[SubmapMapping] Created new submap, now have {submap_id+1} submaps")
        return self.submap_collection

    def need_create_new_submap(self, R, T):
        if self.frame_count == 0:
            return True
        if self.frame_count % self.keyframe_step == 0:
            return True
        return False

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        pass

    def local_to_global(self):
        self.global_map.fuse_submaps(self.submap_collection)

    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
        if self.need_create_new_submap(R, T):
            #In early debug we use framecount as frameid
            self.create_new_submap(self.frame_count, R, T)
        self.submap_collection.recast_depth_to_map(R, T, depthmap, texture, w, h, K, Kcolor)
        self.frame_count += 1
    
    def cvt_TSDF_to_voxels_slice(self, z):
        if len(self.submaps) > 0:
            if self.exporting_global:
                self.global_map.cvt_TSDF_to_voxels_slice(z)
            else:
                self.submap_collection.cvt_TSDF_to_voxels_slice(z)

    def cvt_TSDF_surface_to_voxels(self):
        if len(self.submaps) > 0:
            if self.exporting_global:
                self.global_map.cvt_TSDF_surface_to_voxels()
            else:
                self.submap_collection.cvt_TSDF_surface_to_voxels()
