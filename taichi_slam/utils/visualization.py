import taichi as ti
import numpy as np
from transformations import *

max_drone_num = 100
max_traj_num = 1024*16

@ti.data_oriented
class TaichiSLAMRender:
    def __init__(self, RES_X, RES_Y):
        self.window = window = ti.ui.Window('TaichiSLAM', (RES_X, RES_Y), vsync=True)
        self.pcl_radius = 0.01

        self.canvas = window.get_canvas()
        self.canvas.set_background_color((207/255.0, 243/255.0, 250/255.0))
        self.scene = ti.ui.Scene()
        self.camera = camera = ti.ui.Camera()
        camera.fov(55)
        
        self.par = None
        self.par_color = None
        self.mesh_vertices = None
        self.mesh_color = None
        self.mesh_normals = None

        self.camera_yaw = 0
        self.camera_pitch = -0.5
        self.camera_distance = 3
        self.camera_min_distance = 0.3
        self.camera_lookat = np.array([0., 0., 0.])
        self.camera_pitch_rate = 3.0
        self.camera_yaw_rate = 3.0
        self.camera_move_rate = 3.0
        self.scale_rate = 5
        self.lock_pos_drone = False
        self.slice_z = 0.5
        self.enable_slice_z = False
        self.enable_mesher = False
        self.par_num = None
        
        self.disp_particles = True
        self.disp_mesh = True
        self.set_camera_pose()
        self.mouse_last = None
        self.init_grid()
        self.window.show()
        self.lines = None
        self.lines_color = None
        self.drone_trajs = {}
        self.available_drone_ids = set()
        self.init_drones()

    def init_drones(self):
        self.drone_frame_lines = ti.Vector.field(3, dtype=ti.f32, shape=max_drone_num*6)
        self.drone_frame_colors = ti.Vector.field(3, dtype=ti.f32, shape=max_drone_num*6)
        self.drone_frame_line_width = 5
        self.drone_frame_line_length = 0.2
        self.drone_trajectory_width = 3
        #Init line color as in R, G, B
        for i in range(max_drone_num):
            self.drone_frame_colors[i*6+0] = [1, 0, 0]
            self.drone_frame_colors[i*6+1] = [1, 0, 0]
            self.drone_frame_colors[i*6+2] = [0, 1, 0]
            self.drone_frame_colors[i*6+3] = [0, 1, 0]
            self.drone_frame_colors[i*6+4] = [0, 0, 1]
            self.drone_frame_colors[i*6+5] = [0, 0, 1]
        self.drone_traj_splines = ti.Vector.field(3, dtype=ti.f32, shape=max_traj_num*2)
        self.drone_traj_colors = ti.Vector.field(3, dtype=ti.f32, shape=max_traj_num*2)
        self.drone_traj_pts = 0
            
    def set_camera_pose(self):
        pos = np.array([-self.camera_distance, 0., 0])
        pos = euler_matrix(0, -self.camera_pitch, -self.camera_yaw)[0:3,0:3]@pos + self.camera_lookat

        self.camera.position(pos[0], pos[1], pos[2])
        self.camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        self.camera.up(0., 0., 1.)
        self.scene.set_camera(self.camera)

    @ti.kernel
    def set_drone_trajectory_kernel(self, _c:ti.i32, traj:ti.template(), colors:ti.template(), 
            pos:ti.types.ndarray(), color:ti.types.ndarray()) -> ti.i32:
        for j in ti.static(range(3)):
            traj[_c][j] = pos[0, j]
            colors[_c][j] = color[j]
        count = _c + 1
        ti.loop_config(serialize=True)
        for i in range(1, pos.shape[0]):
            c = ti.atomic_add(count, 2)
            for j in ti.static(range(3)):
                traj[c][j] = pos[i, j]
                traj[c + 1][j] = pos[i, j]
                colors[c][j] = color[j]
                colors[c + 1][j] = color[j]
        return count - 1

    def set_drone_trajectory(self, drone_id, trajectory):
        if trajectory.shape[0] <= 0:
            return
        if drone_id not in self.drone_trajs:
            self.drone_trajs[drone_id] = {
                "color": np.random.rand(3),
                "update": True
            }
        self.available_drone_ids.add(drone_id)
        self.drone_trajs[drone_id]["lines"] = trajectory

    def update_trajs(self):
        c = 0
        for drone_id in self.drone_trajs:
            if "lines" in self.drone_trajs[drone_id] and self.drone_trajs[drone_id]["lines"] is not None:
                traj = self.drone_trajs[drone_id]["lines"]
                color = self.drone_trajs[drone_id]["color"]
                c = self.set_drone_trajectory_kernel(c, self.drone_traj_splines, self.drone_traj_colors, traj, color)
        self.drone_traj_pts = c

    def options(self):
        window = self.window

        window.GUI.begin("Options", 0.05, 0.45, 0.2, 0.4)
        self.pcl_radius = window.GUI.slider_float("particles radius ",
                                            self.pcl_radius, 0.005, 0.03)
        self.lock_pos_drone = window.GUI.checkbox("Look Drone", self.lock_pos_drone)
        self.enable_mesher = window.GUI.checkbox("Enable Mesher", self.enable_mesher)
        self.slice_z = window.GUI.slider_float("slice z",
                                            self.slice_z, 0, 2)
        self.enable_slice_z = window.GUI.checkbox("Show Z Slice", self.enable_slice_z)
        self.disp_particles = window.GUI.checkbox("Particle", self.disp_particles)
        self.disp_mesh = window.GUI.checkbox("Mesh", self.disp_mesh)
        self.camera_distance = window.GUI.slider_float("camera_distance", 
                self.camera_distance, self.camera_min_distance, 100)
        # self.disp_level = math.floor(window.GUI.slider_float("display level ",
        #                                     self.disp_level, 0, 10))
        window.GUI.end()
    
    def set_particles(self, par, color, num=None):
        self.par = par
        self.par_color = color
        self.par_num = num
    
    def set_lines(self, lines, color=None, num=None):
        if type(lines) is np.ndarray:
            #Initialize lines of taichi vector field
            self.lines_color = ti.Vector.field(3, dtype=ti.f32, shape=lines.shape[0])
            self.lines = ti.Vector.field(3, dtype=ti.f32, shape=lines.shape[0])
            self.lines.from_numpy(lines)
            self.lines_color.from_numpy(color)
            self.line_vertex_num = num
        else:
            self.lines = lines
            self.lines_color = color
            self.line_vertex_num = num
    
    def set_mesh(self, mesh, color, normals=None, indices=None, mesh_num=None):
        self.mesh_vertices = mesh
        self.mesh_color = color
        self.mesh_normals = normals
        self.mesh_indices = indices
        self.mesh_num = mesh_num

    def set_drone_pose(self, drone_id, R, T):
        x = np.array([self.drone_frame_line_length, 0, 0])
        y = np.array([0, self.drone_frame_line_length, 0])
        z = np.array([0, 0, self.drone_frame_line_length])
        x = R@x + T
        y = R@y + T
        z = R@z + T
        self.drone_frame_lines[drone_id*6+0] = T
        self.drone_frame_lines[drone_id*6+1] = x
        self.drone_frame_lines[drone_id*6+2] = T
        self.drone_frame_lines[drone_id*6+3] = y
        self.drone_frame_lines[drone_id*6+4] = T
        self.drone_frame_lines[drone_id*6+5] = z
    
    def drone_num(self):
        return len(self.available_drone_ids)

    def handle_events(self):
        win = self.window
        x, y = win.get_cursor_pos()
        if self.mouse_last is None:
            self.mouse_last = win.get_cursor_pos()
        x_s = self.mouse_last[0]
        y_s = self.mouse_last[1]
        if win.is_pressed(ti.ui.LMB):
            self.camera_pitch += self.camera_pitch_rate*(y-y_s)
            self.camera_yaw += self.camera_yaw_rate*(x-x_s)
        if win.is_pressed(ti.ui.MMB):
            R = euler_matrix(0, -self.camera_pitch, -self.camera_yaw)[0:3,0:3]
            move = self.camera_move_rate*self.camera_distance*np.array([0, x-x_s, -(y-y_s)])
            self.camera_lookat += R@move
        if win.is_pressed(ti.ui.RMB):
            move = self.scale_rate*(y-y_s)
            self.camera_distance += move
            if self.camera_distance < self.camera_min_distance:
                self.camera_distance = self.camera_min_distance
        self.camera_lookat[2] = 0.0 #Lock on XY
        self.mouse_last = (x, y)

    def rendering(self):
        self.handle_events()
        self.set_camera_pose()
        self.add_env()
        scene = self.scene
        if self.disp_particles and self.par is not None:
            scene.particles(self.par, per_vertex_color=self.par_color, radius=self.pcl_radius, index_count=self.par_num)
        if self.disp_mesh and self.mesh_vertices is not None:
            scene.mesh(self.mesh_vertices, indices=self.mesh_indices, normals=self.mesh_normals,
               index_count=self.mesh_num, per_vertex_color=self.mesh_color, two_sided=True)

        # #Some additional lines        
        if self.lines is not None:
            scene.lines(self.lines, self.grid_width*5, per_vertex_color=self.lines_color, vertex_count=self.line_vertex_num)
        
        #Drone frame
        scene.lines(self.drone_frame_lines, self.drone_frame_line_width, per_vertex_color=self.drone_frame_colors, vertex_count=self.drone_num()*6)
        # Drone trajectory
        self.update_trajs()
        scene.lines(self.drone_traj_splines, self.drone_trajectory_width, per_vertex_color=self.drone_traj_colors, vertex_count=self.drone_traj_pts)

        self.canvas.scene(scene)
        self.options()
        self.window.show()
    
    def add_env(self):
        scene = self.scene
        scene.ambient_light((1.0, 1.0, 1.0))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.lines(self.grid_lines, self.grid_width, per_vertex_color=self.grid_colors)
            
    def init_grid(self, lines=1024*64, max_lines_x=50, min_lines_y=-50, max_lines_y=50):
        self.grid_lines = ti.Vector.field(3, dtype=ti.f32, shape=lines*2)
        self.grid_colors = ti.Vector.field(4, dtype=ti.f32, shape=lines*2)
        self.grid_width = 1.5
        #Init planar grid
        line_cnt = 0
        for i in range(-max_lines_x, max_lines_x):
            if i == 0:
                self.grid_lines[line_cnt*2] = [i, min_lines_y, 0]
                self.grid_lines[line_cnt*2 + 1] = [i, 0, 0]
                self.grid_colors[line_cnt*2] = [0.5, 0.5, 0.5, 0.5]
                self.grid_colors[line_cnt*2 + 1] = [0.5, 0.5, 0.5, 0.5]
            else:
                self.grid_lines[line_cnt*2] = [i, min_lines_y, 0]
                self.grid_lines[line_cnt*2 + 1] = [i, max_lines_y, 0]
                self.grid_colors[line_cnt*2] = [0.5, 0.5, 0.5, 0.5]
                self.grid_colors[line_cnt*2 + 1] = [0.5, 0.5, 0.5, 0.5]
            line_cnt += 1
        for j in range(min_lines_y, max_lines_y):
            if j == 0:
                self.grid_lines[line_cnt*2] = [-max_lines_x, j, 0]
                self.grid_lines[line_cnt*2 + 1] = [0, j, 0]
                self.grid_colors[line_cnt*2] = [0.5, 0.5, 0.5, 0.5]
                self.grid_colors[line_cnt*2 + 1] = [0.5, 0.5, 0.5, 0.5]
            else:
                self.grid_lines[line_cnt*2] = [-max_lines_x, j, 0]
                self.grid_lines[line_cnt*2+1] = [max_lines_x, j, 0]
                self.grid_colors[line_cnt*2] = [0.5, 0.5, 0.5, 0.5]
                self.grid_colors[line_cnt*2 + 1] = [0.5, 0.5, 0.5, 0.5]
            line_cnt += 1
        #RGB Axis
        # X axis
        self.grid_lines[line_cnt*2] = [0, 0, 0]
        self.grid_lines[line_cnt*2+1] = [max_lines_x, 0, 0]
        self.grid_colors[line_cnt*2] = [1, 0, 0, 1.0]
        self.grid_colors[line_cnt*2 + 1] = [1, 0, 0, 1.0]
        line_cnt += 1
        # Y axis
        self.grid_lines[line_cnt*2] = [0, 0, 0]
        self.grid_lines[line_cnt*2+1] = [0, max_lines_y, 0]
        self.grid_colors[line_cnt*2] = [0, 1, 0, 1.0]
        self.grid_colors[line_cnt*2 + 1] = [0, 1, 0, 1.0]
        line_cnt += 1
        # Z axis
        self.grid_lines[line_cnt*2] = [0, 0, 0]
        self.grid_lines[line_cnt*2+1] = [0, 0, max_lines_y]
        self.grid_colors[line_cnt*2] = [0, 0, 1, 1.0]
        self.grid_colors[line_cnt*2 + 1] = [0, 0, 1, 1.0]
        line_cnt += 1
        