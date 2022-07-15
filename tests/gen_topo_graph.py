#!/usr/bin/env python
import sys, os
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.mapping import *
from taichi_slam.utils.visualization import *
import time

def test(mapping, start_pt, render: TaichiSLAMRender):
    print("Start test topo graph generation")
    print("test detect collision")
    topo = TopoGraphGen(mapping)
    # start_pt = [3.2, 1.2, 1.0]
    topo.test_detect_collisions(start_pt)
    topo.node_expansion(start_pt, False)
    render.set_mesh(topo.tri_vertices, topo.tri_colors, mesh_num=topo.num_triangles[None])

if __name__ == "__main__":
    np.random.seed(0)
    ti.init(arch=ti.cuda)
    # densemap = DenseTSDF.loadMap(os.path.dirname(__file__) + "/../data/ri_tsdf.npy")
    densemap = DenseTSDF.loadMap("/home/xuhao/output/test_map.npy")
    densemap.cvt_TSDF_surface_to_voxels()
    render = TaichiSLAMRender(1920, 1080)
    render.pcl_radius = densemap.voxel_size/2
    start_pt = np.array([1.0, -5, 1.0], dtype=np.float32)
    # start_pt = [0.0, -0, 1.0]
    test(densemap, start_pt, render)
    while True:
        try:
            if render.enable_slice_z:
                densemap.cvt_TSDF_to_voxels_slice(render.slice_z, clear_last=True)
            else:
                densemap.cvt_TSDF_surface_to_voxels()
            densemap.export_TSDF_xyz[densemap.num_TSDF_particles[None]] = start_pt
            densemap.export_color[densemap.num_TSDF_particles[None]] = ti.Vector([0, 0., 0])
            render.set_particles(densemap.export_TSDF_xyz, densemap.export_color, densemap.num_TSDF_particles[None] + 1)
            render.rendering()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break