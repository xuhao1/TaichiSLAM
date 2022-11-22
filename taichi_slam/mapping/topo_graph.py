import taichi as ti
from .mapping_common import *
from scipy.spatial import ConvexHull
import time

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
    
vec3f = ti.types.vector(3, float)

@ti.dataclass
class Facelet:
    normal: vec3f
    edge1: vec3f
    edge2: vec3f
    poly_idx: ti.int32
    facelet_idx: ti.int32
    v0: vec3f
    v1: vec3f
    v2: vec3f
    is_frontier: ti.int32
    center: vec3f
    assigned: ti.i32

    @ti.func
    def init(self, poly_idx, facelet_idx, v0, v1, v2, naive_norm):
        self.poly_idx = poly_idx
        self.edge1 = v1 - v0
        self.edge2 = v2 - v0
        self.center = (v0 + v1 + v2) / 3
        self.normal = self.edge1.cross(self.edge2).normalized()
        if self.normal.dot(naive_norm) < 0:
            self.normal = -self.normal
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.facelet_idx = facelet_idx
        self.assigned = False
        return self

    @ti.func
    def rayTriangleIntersect(self, P, w):
        q = w.cross(self.edge2)
        a = self.edge1.dot(q)
        succ = False
        t = 0.0
        # print("a:", a)
        if ti.abs(a) > 0.00001:
            s = (P-self.v0)/a
            r = s.cross(self.edge1)
            b0 = s.dot(q)
            b1 = r.dot(w)
            b2 = 1.0 - b0 - b1
            t = self.edge2.dot(r)
            succ = True
            # print("b0", b0, "b1", b1, "b2", b2, "t", t)
            if b0 < 0.0 or b1 < 0.0 or b2 < 0.0:
                succ = False
        return succ, t
    
    @ti.func
    def to_str(self):
        return f"Poly {self.poly_idx} Tri {self.facelet_idx} normal: {self.normal[0]:.2f} {self.normal[1]:.2f}  {self.normal[2]:.2f} center: {self.center[0]:.2f} {self.center[1]:.2f} {self.center[2]:.2f}"

@ti.dataclass
class Frontier:
    frontier_idx: ti.i32
    master_idx: ti.i32
    avg_center: vec3f
    outwards_unit_normal: vec3f
    projected_center: vec3f
    projected_normal: vec3f
    next_node_pos: vec3f
    next_node_initial: vec3f
    is_valid : ti.i32
    @ti.func
    def init(self, master_idx, frontier_idx, avg_center, outwards_unit_normal):
        self.master_idx = master_idx
        self.frontier_idx = frontier_idx
        self.avg_center = avg_center
        self.outwards_unit_normal = outwards_unit_normal
        self.is_valid = False

@ti.dataclass
class Gate:
    gate_idx: ti.i32
    master_idx: ti.i32
    frontier_idx: ti.i32
    avg_center: vec3f
    outwards_unit_normal: vec3f
    projected_center: vec3f
    projected_normal: vec3f
    next_node_pos: vec3f
    
    @ti.func
    def init(self, gate_idx, avg_center, outwards_unit_normal):
        self.gate_idx = gate_idx
        self.avg_center = avg_center
        self.outwards_unit_normal = outwards_unit_normal

@ti.dataclass
class MapNode:
    start_facelet_idx: ti.i32
    end_facelet_idx: ti.i32
    center: vec3f
    idx: ti.i32
    master_idx: ti.i32
    
    @ti.func
    def init(self, idx, master_idx, start_facelet_idx, end_facelet_idx, center):
        self.idx = idx
        self.master_idx = master_idx
        self.start_facelet_idx = start_facelet_idx
        self.end_facelet_idx = end_facelet_idx
        self.center = center

@ti.data_oriented
class TopoGraphGen:
    mapping: BaseMap
    def __init__(self, mapping: BaseMap, coll_det_num = 128, max_raycast_dist=2, 
            max_facelets=1024*1024, 
            thres_size=0.5, transparent=0.7, transparent_frontier=0.6, 
            frontier_creation_threshold=0.5,
            frontier_verify_threshold=0.5, 
            frontier_backward_check=-0.2,
            frontier_combine_angle_threshold=40):
        self.mapping = mapping
        self.coll_det_num = coll_det_num
        self.generate_uniform_sample_points(coll_det_num)
        self.max_raycast_dist = max_raycast_dist
        self.init_fields(max_facelets, coll_det_num)
        self.thres_size = thres_size
        self.init_colormap(max_facelets, transparent)
        self.frontier_creation_threshold = frontier_creation_threshold
        self.frontier_verify_threshold = frontier_verify_threshold
        self.transparent_frontier = transparent_frontier
        self.frontier_normal_dot_threshold = np.cos(np.deg2rad(frontier_combine_angle_threshold))
        self.check_frontier_small_distance = 0.1
        self.frontier_backward_check = frontier_backward_check
    
    def init_colormap(self, max_facelets, transparent):
        self.colormap = ti.Vector.field(4, float, shape=max_facelets)
        colors = np.random.rand(max_facelets, 4).astype(np.float32)
        colors[:, 3] = transparent
        self.colormap.from_numpy(colors)
        self.debug_frontier_color = ti.Vector([1.0, 1.0, 1.0, 0.6])

    def init_fields(self, max_facelets, coll_det_num, facelet_nh_search_queue_size=1024, max_map_node=4*1024):
        self.tri_vertices = ti.Vector.field(3, ti.f32, shape=max_facelets*3)
        self.tri_colors = ti.Vector.field(4, ti.f32, shape=max_facelets*3)
        self.facelets = Facelet.field(shape=(max_facelets))
        self.frontiers = Frontier.field(shape=(max_facelets))
        self.nodes = MapNode.field(shape=(max_map_node))

        self.num_facelets = ti.field(dtype=ti.i32, shape=())
        self.num_nodes = ti.field(dtype=ti.i32, shape=())
        self.num_frontiers = ti.field(dtype=ti.i32, shape=())
        self.start_point = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.facelet_nh_search_queue = ti.field(ti.i32, shape=facelet_nh_search_queue_size)
        self.facelet_nh_search_idx = ti.field(dtype=ti.i32, shape=())
        self.facelet_nh_search_queue_size = ti.field(dtype=ti.i32, shape=())
        self.search_frontiers_idx = ti.field(dtype=ti.i32, shape=())
        self.num_facelets[None] = 0
        self.num_nodes[None] = 0
        self.num_frontiers[None] = 0
        self.search_frontiers_idx[None] = 0
        
        self.white_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.white_num = ti.field(dtype=ti.i32, shape=())
        self.black_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.black_unit_list = ti.Vector.field(3, ti.f32, shape=coll_det_num)
        self.black_len_list = ti.field(ti.f32, shape=coll_det_num)
        self.neighbor_node_ids = ti.field(ti.i32, shape=coll_det_num)
        self.neighbor_node_num = ti.field(ti.i32, shape=())
        self.black_num = ti.field(dtype=ti.i32, shape=())
        self.black_num[None] = 0
        self.white_num[None] = 0
        self.neighbor_node_num[None] = 0

        self.edges = ti.Vector.field(3, ti.f32, shape=max_facelets*64)
        self.edge_color = ti.Vector.field(3, ti.f32, shape=max_facelets*64)
        self.edge_num = ti.field(dtype=ti.i32, shape=())
        self.edge_num[None] = 0
        self.connected_nodes_root = ti.root.dense(ti.i, (max_map_node)).pointer(ti.j, (max_map_node))
        self.connected_nodes = ti.field(dtype=ti.i32)
        self.connected_nodes_root.place(self.connected_nodes)
    
    def reset(self):
        self.num_facelets[None] = 0
        self.num_nodes[None] = 0
        self.num_frontiers[None] = 0
        self.search_frontiers_idx[None] = 0
        self.facelet_nh_search_idx[None] = 0
        self.facelet_nh_search_queue_size[None] = 0
        self.edge_num[None] = 0
        self.black_num[None] = 0
        self.white_num[None] = 0
        self.connected_nodes_root.deactivate_all()
    
    def generate_uniform_sample_points(self, npoints):
        phi = np.pi * (3 - np.sqrt(5))
        ret = []
        for i in range(npoints):
            y = 1 - 2 * (i / (npoints - 1))
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            ret.append([x, y, z])
        ret = np.array(ret, dtype=np.float32)
        self.sample_dirs = ti.Vector.field(3, dtype=ti.f32, shape=npoints)
        for i in range(npoints):
            self.sample_dirs[i] = ti.Vector(ret[i, :])

    def generate_random_sample_points(self, npoints):
        self.sample_dirs = ti.Vector.field(3, dtype=ti.f32, shape=npoints)
        vec = np.random.randn(3, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        for i in range(npoints):
            self.sample_dirs[i] = ti.Vector(vec[:, i])
    
    def node_expansion_benchmark(self, start_pt, show=False, run_num=100):
        import time 
        self.start_point[None] = ti.Vector(start_pt, ti.f32)
        s = time.time()
        for i in range(run_num):
            self.detect_collisions()
        print(f"avg detect_collisions time {(time.time()-s)*1000/run_num:.3f}ms")
        s = time.time()
        for i in range(run_num):
            self.generate_poly_on_blacks(start_pt, show)
        print(f"avg gen convex cost time {(time.time()-s)*1000/run_num:.3f}ms")

    def node_expansion(self, start_pt, show=False, last_node_idx = -1):
        self.start_point[None] = start_pt
        s = time.time()
        if self.detect_collisions():
            # print(f"detect_collisions time {(time.time()-s)*1000:.1f}ms")
            # s = time.time()
            self.generate_poly_on_blacks(start_pt, show, last_node_idx)
            # print(f"generate_poly_on_blacks time {(time.time()-s)*1000:.1f}ms")
        # else:
        #     print(f"detect_collisions time {(time.time()-s)*1000:.1f}ms")
    @ti.kernel
    def verify_frontier(self, frontier_idx: ti.i32) -> ti.i32:
        normal = self.frontiers[frontier_idx].projected_normal
        proj_center = self.frontiers[frontier_idx].projected_center + normal * ti.static(self.check_frontier_small_distance)
        succ, t, col_pos, _len, node_idx = self.raycast(proj_center, normal, self.max_raycast_dist*2)
        # print("verify_frontier idx:", frontier_idx, "succ", succ, "type", t, "proj_center", proj_center, "col_pos", col_pos, "len", _len, "node_idx", node_idx)
        if succ and _len < self.frontier_verify_threshold:
            self.frontiers[frontier_idx].is_valid = False
        else:       
            proj_center = self.frontiers[frontier_idx].projected_center - normal * ti.static(self.check_frontier_small_distance)
            succ2, col_pos2, _len2, node_idx2 = self.detect_collision_facelets(proj_center, 
                normal, self.frontier_verify_threshold, self.frontier_backward_check, self.frontiers[frontier_idx].master_idx)
            # print("     step2", succ2, "proj_center", self.frontiers[frontier_idx].projected_center, "col_pos", col_pos2, "len", _len2, "node_idx", node_idx2, "master", self.frontiers[frontier_idx].master_idx)
            if succ2 and _len2 < self.frontier_verify_threshold:
                self.frontiers[frontier_idx].is_valid = False
            else:
                if not succ or succ2 and _len2 < _len:
                    _len = _len2
                self.frontiers[frontier_idx].is_valid = True
                self.frontiers[frontier_idx].next_node_initial = self.frontiers[frontier_idx].projected_center + \
                    self.frontiers[frontier_idx].projected_normal*_len/2
                #Check if near to existing nodes
                # for i in range(self.num_nodes[None]):
                #     if (self.nodes[i].center - self.frontiers[frontier_idx].next_node_initial).norm() < self.node_near_threshold:
                #         self.frontiers[frontier_idx].is_valid = False
                        # print("near to existing node", i, "center", self.nodes[i].center, "next_node_initial", self.frontiers[frontier_idx].next_node_initial)
        # print("    idx: ", frontier_idx, "is_valid", self.frontiers[frontier_idx].is_valid)
        return self.frontiers[frontier_idx].is_valid 
        
    def generate_topo_graph(self, start_pt, max_nodes=100, show=False):
        self.node_expansion(start_pt, show)
        while self.search_frontiers_idx[None] < self.num_frontiers[None] and self.search_frontiers_idx[None] < max_nodes:
            s = time.time()
            if self.verify_frontier(self.search_frontiers_idx[None]):
                # print(f"verify_frontier {self.search_frontiers_idx[None]} time {(time.time()-s)*1000:.1f}ms")
                frontier = self.frontiers[self.search_frontiers_idx[None]]
                self.node_expansion(frontier.next_node_initial, show, last_node_idx=frontier.master_idx)
                # print(f"node_expansion {self.search_frontiers_idx[None]} time {(time.time()-s)*1000:.1f}ms")
            self.search_frontiers_idx[None] += 1
        return self.num_nodes[None]

    def generate_mesh_from_hull(self, hull, start_pt):
        if not isinstance(start_pt, np.ndarray):
            start_pt = start_pt.to_numpy()
        lens = self.black_len_list.to_numpy()[0:self.black_num[None]]
        vertices = hull.points*lens[:, None]
        vertices = np.apply_along_axis(lambda x:x+start_pt, 1, vertices)
        neighbors = hull.neighbors
        return vertices[hull.simplices], neighbors

    def generate_poly_on_blacks(self, start_pt, show=False, last_node_idx=-1):
        black_dirs = self.black_unit_list.to_numpy()[0:self.black_num[None]]
        hull = ConvexHull(black_dirs)
        mesh, neighbors  = self.generate_mesh_from_hull(hull, start_pt)
        s = time.time()
        self.add_mesh(mesh, neighbors, last_node_idx)
        # print(f"add_mesh time {(time.time()-s)*1000:.1f}ms")
        if show:
            show_convex(hull)
            show_mesh(mesh)
    
    @ti.func
    def add_edge(self, a, b, color_a, color_b):
        edge_pt_idx = ti.atomic_add(self.edge_num[None], 2)
        self.edges[edge_pt_idx] = a
        self.edges[edge_pt_idx+1] = b
        self.edge_color[edge_pt_idx] = color_a
        self.edge_color[edge_pt_idx+1] = color_b
    
    @ti.func
    def detect_facelet_frontier(self, facelet):
        #First check if center has obstacle.
        is_frontier = True
        mapping = self.mapping
        if mapping.is_near_pos_occupy(facelet.center, 0) or mapping.is_pos_unobserved(facelet.center):
            is_frontier = False
        else:
            start_raycast_pos = facelet.center + facelet.normal*mapping.voxel_size
            if mapping.is_pos_occupy(start_raycast_pos) or mapping.is_pos_unobserved(facelet.center):
                is_frontier = False
            else:
                max_dist = ti.static(self.frontier_creation_threshold)
                succ, t, col_pos, _len, node_idx = self.raycast(start_raycast_pos, facelet.normal, max_dist)
                if succ and t == 1:
                    self.neighbor_node_ids[ti.atomic_add(self.neighbor_node_num[None], 1)] = node_idx
                if succ:
                    is_frontier = False
        return is_frontier
    
    @ti.func
    def construct_frontier(self, node_idx, idx_start_facelet):
        frontier_idx = ti.atomic_add(self.num_frontiers[None], 1)
        # print("\nconstruct_frontier with ", self.facelet_nh_search_queue_size[None], " facelets")
        center = ti.Vector([0., 0., 0.], ti.f32)
        normal = ti.Vector([0., 0., 0.], ti.f32)
        for i in range(self.facelet_nh_search_queue_size[None]):
            facelet_idx = self.facelet_nh_search_queue[i] + idx_start_facelet # defined in generate_facelets
            center += self.facelets[facelet_idx].center
            normal += self.facelets[facelet_idx].normal
        center /= self.facelet_nh_search_queue_size[None]
        normal = (normal/self.facelet_nh_search_queue_size[None]).normalized()
        self.frontiers[frontier_idx].init(node_idx, frontier_idx, center, normal)
        #Now we raycast the center to facelets to find proj_center.
        succ = False
        t = 0.0
        projected_normal = ti.Vector([0., 0., 0.], ti.f32)
        for i in range(self.facelet_nh_search_queue_size[None]):
            facelet_idx = self.facelet_nh_search_queue[i] + idx_start_facelet 
            succ, t = self.facelets[facelet_idx].rayTriangleIntersect(center, normal)
            projected_normal = self.facelets[facelet_idx].normal
            if succ:
                break
        if succ:
            proj_center = center + t*normal
            self.frontiers[frontier_idx].projected_center = proj_center
            self.frontiers[frontier_idx].projected_normal = projected_normal
            # self.add_edge(proj_center, proj_center+projected_normal*0.5, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 1.0, 0.0]))
        else:
            # print("Failed to raycast center set facelets to different color: retract frontier", self.colormap[frontier_idx + 10])
            # for i in range(self.facelet_nh_search_queue_size[None]):
            #     facelet_idx = self.facelet_nh_search_queue[i] + idx_start_facelet 
            #     for j in range(ti.static(3)):
            #         self.tri_colors[facelet_idx*3 + j] = self.colormap[frontier_idx + 10]
            self.num_frontiers[None] -= 1

    @ti.kernel
    def add_mesh(self, mesh: ti.types.ndarray(), neighbors:ti.types.ndarray(), last_node_idx: ti.i32):
        num_facelets = mesh.shape[0]
        facelet_start_idx = ti.atomic_add(self.num_facelets[None], num_facelets)
        center_pos = ti.Vector([0., 0., 0.], ti.f32)
        center_count = 0.0
        for i in range(num_facelets):
            facelet_idx = i + facelet_start_idx #To avoid parallel make the indice wrong
            for j in range(ti.static(3)):
                self.tri_vertices[facelet_idx*3 + j] = ti.Vector(
                        [mesh[i, j, 0], mesh[i, j, 1], mesh[i, j, 2]], ti.f32)
            v0 = self.tri_vertices[facelet_idx*3]
            v1 = self.tri_vertices[facelet_idx*3 + 1]
            v2 = self.tri_vertices[facelet_idx*3 + 2]
            center_pos += v0 + v1 + v2
            center_count += 3.0
            naive_norm = (v0 + v1 + v2 - 3.0*self.start_point[None]).normalized()
            self.facelets[facelet_idx].init(self.num_nodes[None], facelet_idx, v0, v1, v2, naive_norm)
            self.facelets[facelet_idx].is_frontier = self.detect_facelet_frontier(self.facelets[facelet_idx])
            #For visualization
            for j in range(ti.static(3)):
                self.tri_colors[facelet_idx*3 + j] = self.colormap[self.num_nodes[None]]
                if self.facelets[facelet_idx].is_frontier:
                    self.tri_colors[facelet_idx*3 + j][3] = ti.static(self.transparent_frontier)
        new_node_center = center_pos/center_count
        self.nodes[self.num_nodes[None]].init(self.num_nodes[None], last_node_idx, facelet_start_idx, facelet_start_idx + num_facelets, new_node_center)
        if last_node_idx >= 0:
            self.add_edge(self.nodes[last_node_idx].center, new_node_center, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]))
            self.connected_nodes[self.num_nodes[None], last_node_idx] = 1
            self.connected_nodes[last_node_idx, self.num_nodes[None]] = 1
        ti.loop_config(serialize=True)
        for i in range(self.neighbor_node_num[None]):
            neigh_idx = self.neighbor_node_ids[i]
            if self.connected_nodes[self.num_nodes[None], neigh_idx] == 0:
                self.connected_nodes[self.num_nodes[None], neigh_idx] = 1
                self.connected_nodes[neigh_idx, self.num_nodes[None]] = 1
                self.add_edge(self.nodes[neigh_idx].center, new_node_center, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]))
        #Construct facelet from neighbors relationship
        ti.loop_config(serialize=True)
        for i in range(facelet_start_idx, facelet_start_idx+num_facelets):
            idx = i - facelet_start_idx  #This idx is define in hull
            if not self.facelets[i].assigned and self.facelets[i].is_frontier:
                # print("\nfacelet ", self.facelets[i].facelet_idx, "idx", idx, " not assigned, use as seed")
                self.facelet_nh_search_idx[None] = 0
                self.facelet_nh_search_queue_size[None] = 1
                self.facelet_nh_search_queue[0] = idx 
                normal = self.facelets[i].normal
                while self.facelet_nh_search_idx[None] < self.facelet_nh_search_queue_size[None]:
                    search_idx = ti.atomic_add(self.facelet_nh_search_idx[None], 1)
                    _idx = self.facelet_nh_search_queue[search_idx] #This idx is define in hull
                    facelet_idx = _idx + facelet_start_idx
                    self.facelets[facelet_idx].assigned = True
                    for j in range(ti.static(3)):
                        #Check if neighor is assigned and frontier
                        idx_neighbor = neighbors[_idx, j] + facelet_start_idx
                        if self.facelets[idx_neighbor].is_frontier and not self.facelets[idx_neighbor].assigned and \
                                normal.dot(self.facelets[idx_neighbor].normal) > ti.static(self.frontier_normal_dot_threshold):
                            self.facelet_nh_search_queue[ti.atomic_add(self.facelet_nh_search_queue_size[None], 1)] = neighbors[_idx, j]
                            # print("neighbor", neighbors[_idx, j], "idx", idx_neighbor, "v0", self.facelets[idx_neighbor].v0, "v1", self.facelets[idx_neighbor].v1, "v2", self.facelets[idx_neighbor].v2)
                # Now we can construct the frontier from these facelets
                # It uses logs in facelet_nh_search_queue to find the facelets
                self.construct_frontier(self.num_nodes[None], facelet_start_idx)
        self.num_nodes[None] = self.num_nodes[None] + 1

    @ti.kernel
    def detect_collisions(self) ->ti.i32:
        pos = self.start_point[None]
        max_raycast_dist = ti.static(self.max_raycast_dist)
        ray_len_black = 0.0
        self.black_num[None] = 0
        self.white_num[None] = 0
        self.neighbor_node_num[None] = 0
        for i in range(ti.static(self.coll_det_num)):
            succ, t, col_pos, _len, node_idx = self.raycast(pos, self.sample_dirs[i], max_raycast_dist)
            if succ:
                index = ti.atomic_add(self.black_num[None], 1)
                self.black_list[index] = col_pos
                self.black_unit_list[index] = self.sample_dirs[i]
                self.black_len_list[index] = _len
                ray_len_black += _len
                if t == 1: #Is connected to other polyhedron
                    _idx = ti.atomic_add(self.neighbor_node_num[None], 1)
                    self.neighbor_node_ids[_idx] = node_idx
            else:
                index = ti.atomic_add(self.white_num[None], 1)
                self.white_list[index] = col_pos
        node_size = ray_len_black/self.black_num[None]
        succ = True
        if self.black_num[None] == 0 or (self.white_num[None] == 0 and node_size < ti.static(self.thres_size)):
            succ = False
        return succ

    @ti.func
    def detect_collision_facelets(self, pos, dir, max_dist:ti.f32, backward_dist:ti.f32=-0.01, skip_idx=-1):
        succ = False
        best_t = max_dist
        best_poly_ind = -1
        for k in range(self.num_nodes[None]):
            if k != skip_idx:
                poly = self.nodes[k]
                if (pos - poly.center).norm() < max_dist + ti.static(self.max_raycast_dist):
                    for i in range(poly.start_facelet_idx, poly.end_facelet_idx):
                        _succ, t = self.facelets[i].rayTriangleIntersect(pos, dir)
                        if _succ and backward_dist < t < best_t:
                            best_t = t
                            best_poly_ind = self.facelets[i].poly_idx
                            succ = True
        pos_poly = pos + dir*best_t
        return succ, pos_poly, best_t, best_poly_ind
                
    @ti.func
    def raycast(self, pos, dir, max_dist, skip_idx=-1):
        mapping = self.mapping
        recast_type = 1 # 0 map 1 poly
        succ_poly, pos_coll, len_coll, poly_ind = self.detect_collision_facelets(pos, dir, max_dist, -0.01, skip_idx)
        # print("raycast: ", pos, dir, max_dist, "\n    poly succ:", succ_poly, "pos", pos_coll, "len", len_coll, "poly", poly_ind)
        max_dist_recast = max_dist
        if succ_poly:
            max_dist_recast = len_coll
        succ_map, pos_col_map, len_map = mapping.raycast(pos, dir, max_dist_recast)
        # print("    raycast map:", succ_map, "col", pos_col_map, "len", len_map)
        if (not succ_poly) or (succ_map and len_map < len_coll):
                pos_coll = pos_col_map
                len_coll = len_map
                recast_type = 0
                succ_poly = succ_map
        # print("    raycast ret succ", succ_poly, "type", recast_type, "pos", pos_coll, "len", len_coll, "poly", poly_ind)
        return succ_poly, recast_type, pos_coll, len_coll, poly_ind

    def test_detect_collisions(self, start_pt):
        self.start_point[None] = ti.Vector(start_pt)
        self.detect_collisions()
