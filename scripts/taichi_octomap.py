# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import tina
import time


ti.init(arch=ti.cpu, debug=True)

RES = 1024
K = 2
R = 9
Rz = 8
N = K**R
Nz = K**Rz
map_scale = 40
map_scale_z = 6
grid_scale = map_scale/N
grid_scale_z = map_scale_z/Nz
max_num_particles = 1000000

Broot = ti.root
B = ti.root
for r in range(R):
    if r < Rz:
        B = B.pointer(ti.ijk, (K, K, K))
    else:
        B = B.pointer(ti.ijk, (K, K, 1))

num_export_particles = ti.field(dtype=ti.i32, shape=())
num_input_points = ti.field(dtype=ti.i32, shape=())

input_R = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
input_T = ti.Vector.field(3, dtype=ti.f32, shape=())
grid_scale_field = ti.Vector.field(3, dtype=ti.f32, shape=())
N_field = ti.Vector.field(3, dtype=ti.f32, shape=())
grid_scale_field = ti.Vector([grid_scale, grid_scale, grid_scale_z])
N_field = ti.Vector([N/2, N/2, Nz/2])
#grid_scale_field[None][0] = grid_scale
#grid_scale_field[None][1] = grid_scale
#grid_scale_field[None][2] = grid_scale_z
# N_field

x = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)
pcl_input = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)

#qt.parent is the deepest of bitmasked
qt = ti.field(ti.f32)
B.place(qt)

print(f'The map voxel is:[{N}x{N}x{Nz}] map scale:[{map_scale}mx{map_scale}mx{map_scale_z}m], grid scale [{grid_scale:3.3f}x{grid_scale:3.3f}x{grid_scale_z:3.3f}]')


@ti.kernel
def random_init_octo(pts: ti.template()):
    for i in range(pts):
        x_ = ti.random(dtype = int)%N
        y_ = ti.random(dtype = int)%N
        z_ = ti.random(dtype = int)%Nz
        qt[x_, y_, z_] = 1

@ti.kernel
def get_voxel_to_particles(level: ti.template()):
    # Number for level
    n = K**(R-level)
    level_grid_scale = K**level
    num_export_particles[None] = 0

    for i, j, k in qt.parent(level+1):
        index = ti.atomic_add(num_export_particles[None], 1)
        x[index][0] = i*grid_scale
        x[index][1] = j*grid_scale
        x[index][2] = k*grid_scale

@ti.kernel
def recast_pcl_to_map():
    print(num_input_points[None])
    for index in range(num_input_points[None]):
        pt = pcl_input[index]
        _pts = input_R@pt + input_T
        _pts = _pts / grid_scale_field + N_field
        _pts.cast(int)
        qt[_pts] = 1


def render_map_to_particles(pars, pos_, num_particles_, level):
    #print(f"set_particles {num_particles_}")
    pos = pos_[0:num_particles_,:]
    pars.set_particles(pos)
    radius = np.ones(num_particles_)*(K**(level))*grid_scale
    pars.set_particle_radii(radius)
    # color = np.random.rand(num_particles_, 3).astype(np.float32) * 0.8 + 0.2
    color = np.ones((num_particles_, 3)).astype(np.float32)
    pars.set_particle_colors(color)


def handle_render(scene, gui, pars, level):
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == "-":
            level += 1
            if level == R:
                level = R - 1
        elif e.key == "=":
            level -= 1
            if level < 0:
                level = 0
    get_voxel_to_particles(level)
    pos_ = x.to_numpy()
    render_map_to_particles(pars, pos_, num_export_particles[None], level)

    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.text(content=f'Level {level:.2f} num_particles {num_export_particles[None]} incress =; decress -',
            pos=(0, 0.8),
            font_size=40,
            color=0xffffff)
    gui.show()
    return level


if __name__ == '__main__':
    gui = ti.GUI('TaichiOctomap', (RES, RES))
    level = R//2
    scene = tina.Scene(RES)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale*2, theta=-1.0, center=(map_scale/2, map_scale/2, map_scale/2))
    Broot.deactivate_all()

    import rosbag
    import sensor_msgs.point_cloud2 as pc2

    #Level = 0 most detailed
    random_init_octo(1000)

    while gui.running:
        level = handle_render(scene, gui, pars, level)