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

def test(mapping, start_pt, render: TaichiSLAMRender, args):
    print("Start test topo graph generation")
    if args.benchmark:
        benchmark(mapping, start_pt, args.run_num)
    topo = TopoGraphGen(mapping, max_raycast_dist=args.ray, coll_det_num=args.coll_det_num, 
        frontier_combine_angle_threshold=20)
    s = time.time()
    num_nodes = topo.generate_topo_graph(start_pt, max_nodes=100000)
    topo.reset()
    s = time.time()
    num_nodes = topo.generate_topo_graph(start_pt, max_nodes=100000)
    print("Topo graph generated nodes", num_nodes, ", time cost", (time.time() - s)*1000, "ms")
    render.set_mesh(topo.tri_vertices, topo.tri_colors, mesh_num=topo.num_facelets[None])
    render.set_skeleton_graph_edges(topo.edges.to_numpy()[0:topo.edge_num[None]])

if __name__ == "__main__":
    import argparse
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='enable benchmark')
    parser.add_argument('--map', type=str, default=os.path.dirname(__file__) + "/../data/ri_tsdf.npy", 
        help='path of map, .npy file')
    parser.add_argument('--run_num', type=int, default=100, help='number of runs for benchmark')
    parser.add_argument('--start_pt', type=str, default="1.0, 0, 1.0",help='start point')
    parser.add_argument('--ray', type=float, default=1.5, help='max raycast distance')
    parser.add_argument('--coll_det_num', type=int, default=64,help='collision detection number')
    args = parser.parse_args()
    start_pt = np.array([float(x) for x in args.start_pt.split(",")])
    print("Start point: ", start_pt)

    ti.init(arch=ti.cpu, offline_cache=False, device_memory_fraction=0.5)
    densemap = DenseTSDF.loadMap(args.map)
    densemap.disp_floor = -1
    densemap.cvt_TSDF_surface_to_voxels()
    render = TaichiSLAMRender(3000, 2000)
    render.pcl_radius = densemap.voxel_size/2
    test(densemap, start_pt, render, args)
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