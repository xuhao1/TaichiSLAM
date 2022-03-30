import taichi as ti
import numpy as np
from transformations import *

class TaichiSLAMRender:
    def __init__(self, RES_X, RES_Y):
        self.window = window = ti.ui.Window('TaichiSLAM', (RES_X, RES_Y), vsync=True)
        self.pcl_radius = 0.01

        self.canvas = window.get_canvas()
        self.canvas.set_background_color((207/255.0, 243/255.0, 250/255.0))
        self.scene = ti.ui.Scene()
        self.camera = camera = ti.ui.make_camera()
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
        
        self.disp_particles = True
        self.disp_mesh = True

        self.set_camera_pose()

        self.mouse_last = None

        self.window.show()
    
    def set_camera_pose(self):
        pos = np.array([-self.camera_distance, 0., 0])
        pos = euler_matrix(0, -self.camera_pitch, -self.camera_yaw)[0:3,0:3]@pos + self.camera_lookat

        self.camera.position(pos[0], pos[1], pos[2])
        self.camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        self.camera.up(0., 0., 1.)
        self.scene.set_camera(self.camera)

    def options(self):
        window = self.window

        window.GUI.begin("Options", 0.05, 0.45, 0.2, 0.4)
        self.pcl_radius = window.GUI.slider_float("particles radius ",
                                            self.pcl_radius, 0.005, 0.03)
        self.lock_pos_drone = window.GUI.checkbox("Look Drone", self.lock_pos_drone)
        self.slice_z = window.GUI.slider_float("slice z",
                                            self.slice_z, 0, 2)

        self.disp_particles = window.GUI.checkbox("Particle", self.disp_particles)
        self.disp_mesh = window.GUI.checkbox("Mesh", self.disp_mesh)
        self.camera_distance = window.GUI.slider_float("camera_distance", 
                self.camera_distance, self.camera_min_distance, 100)
        # self.disp_level = math.floor(window.GUI.slider_float("display level ",
        #                                     self.disp_level, 0, 10))
        window.GUI.end()
    
    def set_particles(self, par, color):
        self.par = par
        self.par_color = color

    def set_mesh(self, mesh, color, normals=None, indices=None):
        self.mesh_vertices = mesh
        self.mesh_color = color
        self.mesh_normals = normals
        self.mesh_indices = None

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
        
        scene = self.scene
        
        # self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.LMB)

        scene.ambient_light((1.0, 1.0, 1.0))

        if self.disp_particles and self.par is not None:
            scene.particles(self.par, per_vertex_color=self.par_color, radius=self.pcl_radius)
        if self.disp_mesh and self.mesh_vertices is not None:
            scene.mesh(self.mesh_vertices,
               indices=self.mesh_indices,
               normals=self.mesh_normals,
               per_vertex_color=self.mesh_color,
               two_sided=True)
            
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        self.canvas.scene(scene)
        self.options()
        self.window.show()
        