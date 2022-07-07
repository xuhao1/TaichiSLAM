from dense_esdf import DenseESDF


class SubmapMapping:
    def __init__(self, submap_type=DenseESDF, num_submaps=10000, sdf_opts={}):
        sdf_default_opts = {
            'map_scale': [10, 10],
            'voxel_size': 0.05,
            'min_occupy_thres': 3,
            'texture_enabled': False,
            'min_ray_length': 0.3,
            'max_ray_length': 3.0,
            'max_disp_particles': 1000000,
            'block_size': 3,
            'enable_esdf': false
        }
        sdf_default_opts.update(map_opts)
        self.sdf_opts = sdf_opts
        self.submaps = []
        self.frame_count = 0
    
    def create_new_submap(self):
        new_submap = DenseESDF(**self.sdf_opts)
        self.submaps.append(new_submap)
        print(f"[SubmapMapping] Created new submap, now have {len(self.submaps)} submaps")
        return new_submap

    def need_create_new_submap(self, R, T):
        if frame_count == 0:
            return True
        if frame_count % 10 == 0:
            return True
        return False

    def recast_pcl_to_map(self, R, T, xyz_array, rgb_array, n):
        pass

    def recast_depth_to_map(self, R, T, depthmap, texture, w, h, K, Kcolor):
        if self.need_create_new_submap(R, T):
            self.create_new_submap()
        frame_count += 1
