#!/usr/bin/env python
import sys, os
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
import time

if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    densemap = DenseTSDF.loadMap(os.path.dirname(__file__) + "/../data/ri_tsdf.npy")
    densemap.cvt_TSDF_surface_to_voxels()
    render = TaichiSLAMRender(1920, 1080)
    render.pcl_radius = densemap.voxel_size/2
    render.set_particles(densemap.export_TSDF_xyz, densemap.export_color)
    while True:
        try:
            render.rendering()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break