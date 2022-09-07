#!/usr/bin/env python
import sys, os
sys.path.insert(0,os.path.dirname(__file__) + "/../")
from taichi_slam.mapping import DenseTSDF, TopoGraphGen
from taichi_slam.utils.visualization import TaichiSLAMRender
import taichi as ti
import numpy as np
import time

def benchmark(mapping, start_pt, run_num):
    topo = TopoGraphGen(mapping, max_raycast_dist=1.5)
    # start_pt = [3.2, 1.2, 1.0]
    topo.test_detect_collisions(start_pt)
    topo.node_expansion(start_pt, False)
    s = time.time()
    topo.node_expansion_benchmark(start_pt, False, run_num=run_num)
    dt = time.time() - s
    print(f"avg node expansion time: {dt*1000/run_num:.2f}ms")

def test(mapping, start_pt, render: TaichiSLAMRender, run_num=100, enable_benchmark=False):
    print("Start test topo graph generation")
    if enable_benchmark:
        benchmark(mapping, start_pt, run_num)
    topo = TopoGraphGen(mapping, max_raycast_dist=1.5)
    topo.generate_topo_graph(start_pt, max_nodes=10000)
    render.set_mesh(topo.tri_vertices, topo.tri_colors, mesh_num=topo.num_facelets[None])
    render.set_lines(topo.lines_show, topo.lines_color, num=topo.lines_num[None])

if __name__ == "__main__":
    np.random.seed(1)
    ti.init(arch=ti.cuda)
    densemap = DenseTSDF.loadMap(os.path.dirname(__file__) + "/../data/ri_tsdf.npy")
    densemap.cvt_TSDF_surface_to_voxels()
    render = TaichiSLAMRender(1920, 1080)
    render.pcl_radius = densemap.voxel_size/2
    start_pt = np.array([1.0, -5, 1.0], dtype=np.float32)
    # start_pt = [0.0, -0, 1.0]
    test(densemap, start_pt, render, enable_benchmark=False)
    render.camera_lookat = start_pt
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