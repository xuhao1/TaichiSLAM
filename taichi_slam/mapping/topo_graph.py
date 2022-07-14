import taichi as ti
from .mapping_common import *
@ti.data_oriented
class TopoGraphGen:
    def __init__(self, mapping: BaseMap):
        self.mapping = mapping
        self.generate_uniform_sample_points()
    
    def generate_uniform_sample_points(self, npoints):
        self.sample_dirs = ti.Vector.field(3, dtype=ti.f32, shape=())
        vec = np.random.randn(3, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        for i in range(npoints):
            self.sample_dirs[i] = ti.Vector(vec[:, i])
    
    @ti.kernel
    def generate_topo_graph(self):
        pass
    