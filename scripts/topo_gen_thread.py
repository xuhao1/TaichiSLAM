#!/usr/bin/env python3
from taichi_slam.mapping import DenseTSDF, TopoGraphGen
import time
import taichi as ti
import numpy as np
class TopoGen:
    def __init__(self, params_map, params_topo, man_d):
        self.mapping = DenseTSDF(is_global_map=True, **params_map)
        self.topo = TopoGraphGen(self.mapping, **params_topo)
        self.man_d = man_d
    
    def run(self):
        print("Start topo graph generation thread")
        while not self.man_d["exit"]:
            try:
                if self.man_d["update"]:
                    #print all the keys and values
                    self.loadMap(self.man_d["map_data"])
                    self.gen_skeleton_graph()
                    self.man_d["update"] = False
                time.sleep(1)
            except KeyboardInterrupt:
                break
    
    def loadMap(self, map_data):
        TSDF = map_data['TSDF']
        W_TSDF = map_data['W_TSDF']
        color = map_data['color']
        indices = map_data['indices']
        occupy = map_data['occupy']
        self.mapping.reset()
        self.mapping.load_numpy(0, indices, TSDF, W_TSDF, occupy, color)
    
    def gen_skeleton_graph(self):
        start_pt = np.array([1., 0., 0.5])
        self.topo.reset()
        s = time.time()
        num_nodes = self.topo.generate_topo_graph(start_pt, max_nodes=100000)
        print(f"[Topo] Number of polygons: {num_nodes} start pt {start_pt} t: {(time.time()-s)*1000:.1f}ms")
        self.export_topo_graph()
    
    def export_topo_graph(self):
        lines = self.topo.lines_show.to_numpy()[0:self.topo.lines_num[None]]
        colors = self.topo.lines_color.to_numpy()[0:self.topo.lines_num[None]]
        self.man_d["topo_graph_viz"] = {"lines": lines, "colors": colors}

def TopoGenThread(params, man_d):
    if params["use_cuda"]:
        ti.init(arch=ti.cuda, dynamic_index=True, offline_cache=True, packed=True, debug=False)
    else:
        ti.init(arch=ti.cpu, dynamic_index=True, offline_cache=True, packed=True, debug=False)
    print("TopoGenThread: params = ", params, man_d)
    topo = TopoGen(params["sdf_params"], params["skeleton_graph_gen_opts"], man_d)
    topo.run()