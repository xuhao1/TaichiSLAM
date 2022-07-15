from chardet import detect
import taichi as ti
from .mapping_common import *
from scipy.spatial import ConvexHull

def show_mesh(mesh):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pc = art3d.Poly3DCollection(mesh, edgecolor="black")
    ax.add_collection(pc)
    plt.pause(0.5)

def show_convex(hull):
    faces = hull.simplices
    vertices = hull.points
    show_mesh(vertices[faces])
    
@ti.data_oriented
class TopoGraphGen:
    def __init__(self, mapping: BaseMap, coll_det_num = 100, max_raycast_dist=2, max_tri=1000000, thres_size=0.5, transparent=0.8):
        self.mapping = mapping
        self.coll_det_num = coll_det_num
        self.generate_uniform_sample_points(coll_det_num)
        self.max_raycast_dist = max_raycast_dist
        self.init_fields(max_tri, coll_det_num)
        self.thres_size = thres_size
        self.transparent = transparent
        self.init_colormap(max_tri, transparent)
    
    def init_colormap(self, max_tri, transparent):
        self.colormap = ti.Vector.field(4, float, shape=max_tri)
        colors = np.random.rand(max_tri, 4)
        colors[:, 3] = transparent
        self.colormap.from_numpy(colors)

    def init_fields(self, max_tri, coll_det_num):
        self.tri_vertices = ti.Vector.field(3, ti.f32, shape=max_tri*3)
        self.tri_colors = ti.Vector.field(4, ti.f32, shape=max_tri*3)
        self.tri_normals = ti.Vector.field(3, ti.f32, shape=max_tri)
        self.tri_poly_indices = ti.field(dtype=ti.i32, shape=max_tri)
        self.num_triangles = ti.field(dtype=ti.i32, shape=())
        self.num_polyhedron = ti.field(dtype=ti.i32, shape=())
        self.num_triangles[None] = 0
        self.num_polyhedron[None] = 0
        
        self.white_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.white_num = ti.field(dtype=ti.i32, shape=())
        self.black_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.black_unit_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.black_len_list = ti.field(ti.f32, shape=coll_det_num)
        self.black_num = ti.field(dtype=ti.i32, shape=())
        self.black_num[None] = 0
        self.white_num[None] = 0

    def generate_uniform_sample_points(self, npoints):
        self.sample_dirs = ti.Vector.field(3, dtype=ti.f32, shape=npoints)
        vec = np.random.randn(3, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        for i in range(npoints):
            self.sample_dirs[i] = ti.Vector(vec[:, i])
    
    def node_expansion(self, start_pt, show=False):
        start_pt = ti.Vector(start_pt, ti.f32)
        if self.detect_collisions(start_pt):
            self.generate_poly_on_blacks(start_pt, show)

    def generate_mesh_from_hull(self, hull, start_pt):
        lens = self.black_len_list.to_numpy()[0:self.black_num[None]]
        vertices = hull.points*lens[:, None]
        vertices = np.apply_along_axis(lambda x:x+start_pt, 1, vertices)
        return vertices[hull.simplices]

    def generate_poly_on_blacks(self, start_pt, show=False):
        black_dirs = self.black_unit_list.to_numpy()[0:self.black_num[None]]
        hull = ConvexHull(black_dirs)
        mesh = self.generate_mesh_from_hull(hull, start_pt)
        self.add_convex(mesh)
        if show:
            show_convex(hull)
            show_mesh(mesh)
    
    @ti.kernel
    def add_convex(self, mesh: ti.types.ndarray()):
        index_poly = ti.atomic_add(self.num_polyhedron[None], 1)
        print(f"add_convex {mesh.shape[0]} cur polys {self.num_polyhedron[None]} tris {self.num_triangles[None]}")
        for i in range(mesh.shape[0]):
            tri_num_old = ti.atomic_add(self.num_triangles[None], 1)
            for j in range(ti.static(3)):
                self.tri_vertices[tri_num_old*3 + j] = ti.Vector(
                        [mesh[i, j, 0], mesh[i, j, 1], mesh[i, j, 2]], ti.f32)
                self.tri_colors[tri_num_old*3 + j] = self.colormap[index_poly]
                
            self.tri_poly_indices[index_poly] = tri_num_old
    
    def generate_convex_on_blacks(self):
        pass

    @ti.kernel
    def generate_topo_graph(self, start_pt: ti.types.ndarray()):
        pass

    @ti.kernel
    def detect_collisions(self, pos:ti.template()) ->ti.i32:
        ti.loop_config(serialize=True, parallelize=False)
        max_raycast_dist = ti.static(self.max_raycast_dist)
        ray_len_black = 0.0
        self.black_num[None] = 0
        self.white_num[None] = 0
        for i in range(ti.static(self.coll_det_num)):
            t, succ, col_pos, _len = self.raycast(pos, self.sample_dirs[i], max_raycast_dist)
            if succ:
                # print("is collision on dir", self.sample_dirs[i], " pt ", col_pos)
                index = ti.atomic_add(self.black_num[None], 1)
                self.black_list[index] = col_pos
                self.black_unit_list[index] = self.sample_dirs[i]
                self.black_len_list[index] = _len
                ray_len_black += _len
            else:
                index = ti.atomic_add(self.white_num[None], 1)
                self.white_list[index] = col_pos
        node_size = ray_len_black/self.black_num[None]
        succ = True
        if self.black_num[None] == 0 or (self.white_num[None] == 0 and node_size < ti.static(self.thres_size)):
            succ = False
        return succ

    @ti.func
    def detect_collision_polys(self, pos, dir, max_dist):
        return False, ti.Vector([0, 0., 0.]), 100000

    @ti.func
    def raycast(self, pos, dir, max_dist):
        mapping = self.mapping
        succ_poly, pos_poly, len_poly = self.detect_collision_polys(pos, dir, max_dist)
        max_dist_recast = max_dist
        if succ_poly:
            max_dist_recast = len_poly
        succ, pos_col, _len = mapping.raycast(pos, dir, max_dist_recast)
        recast_type = 0 # 0 map 1 poly
        if succ_poly:
            #Then we return the nearset
            if len_poly < _len:
                pos_col = pos_poly
                _len = len_poly
                recast_type = 1
        return recast_type, succ, pos_col, _len

    def test_detect_collisions(self, start_pt):
        _start_pt = ti.Vector(start_pt)
        self.detect_collisions(_start_pt)