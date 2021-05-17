# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import tina
import time
from matplotlib import cm


ti.init(arch=ti.cpu, debug=True)

RES = 1024
K = 2
R = 8
Rz = 7
N = K**R
Nz = K**Rz
map_scale = 20
map_scale_z = 10
grid_scale = map_scale/N
grid_scale_z = map_scale_z/Nz
max_num_particles = 1000000
MIN_RECAST_THRES = 2
Broot = ti.root
B = ti.root
TEXTURE_ENABLED = True

for r in range(R):
    if r < Rz:
        B = B.pointer(ti.ijk, (K, K, K))
    else:
        B = B.pointer(ti.ijk, (K, K, 1))

Croot = ti.root
C = ti.root
for r in range(R):
    if r < Rz:
        C = C.pointer(ti.ijk, (K, K, K))
    else:
        C = C.pointer(ti.ijk, (K, K, 1))

num_export_particles = ti.field(dtype=ti.i32, shape=())

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
color = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)

#qt.parent is the deepest of bitmasked
qt = ti.field(ti.i32)
Cqt = ti.Vector.field(3, ti.i32)

B.place(qt)
C.place(Cqt)

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
    num_export_particles[None] = 0
    for i, j, k in qt.parent(level+1):
        if qt[i, j, k] > MIN_RECAST_THRES:
            index = ti.atomic_add(num_export_particles[None], 1)
            if num_export_particles[None] < max_num_particles:
                x[index][0] = i*grid_scale - map_scale/2
                x[index][1] = j*grid_scale - map_scale/2
                x[index][2] = k*grid_scale_z  - map_scale_z/2
                if TEXTURE_ENABLED:
                    color[index] = Cqt[i, j, k]

@ti.kernel
def recast_pcl_to_map(xyz_array: ti.ext_arr(), rgb_array: ti.ext_arr(), n: ti.i32):
    for index in range(n):
        pt = ti.Vector([
            xyz_array[index,0], 
            xyz_array[index,1], 
            xyz_array[index,2]])
        _pts = input_R@pt + input_T
        _pts = _pts / grid_scale_field + N_field
        _pts.cast(int)

        if _pts[0] >= N:
            _pts[0] = N - 1
        if _pts[1] >= N:
            _pts[1] = N - 1
        if _pts[2] >= Nz:
            _pts[2] = Nz - 1

        if _pts[0] < 0:
            _pts[0] = 0
        if _pts[1] < 0:
            _pts[1] = 0
        if _pts[2] < 0:
            _pts[2] = 0

        qt[_pts] += 1

        if TEXTURE_ENABLED:
            for d in ti.ti.static(range(3)):
                Cqt[_pts][d] = rgb_array[index, d]

@ti.kernel
def recast_pcl_to_map_no_project(xyz_array: ti.ext_arr(), n: ti.i32):
    for index in range(n):
        pt = ti.Vector([
            xyz_array[index,0], 
            xyz_array[index,1], 
            xyz_array[index,2]])
        _pts = pt / grid_scale_field + N_field
        _pts.cast(int)
        if _pts[0] >= N:
            _pts[0] = N - 1
        if _pts[1] >= N:
            _pts[1] = N - 1
        if _pts[2] >= Nz:
            _pts[2] = Nz - 1

        if _pts[0] < 0:
            _pts[0] = 0
        if _pts[1] < 0:
            _pts[1] = 0
        if _pts[2] < 0:
            _pts[2] = 0

        qt[_pts] += 1

def render_map_to_particles(pars, pos_, colors, num_particles_, level):
    #print(f"set_particles {num_particles_}")
    pos = pos_[0:num_particles_,:]

    #Forward Left Up to screen
    #pos[:,[0, 1, 2]] = pos[:,[1,0,2]]
    #pos[:,0] = - pos[0:,0]
    #pos[:,2] = - pos[0:,2]
    if not TEXTURE_ENABLED:
        max_z = np.max(pos[:,2])
        min_z = np.min(pos[:,2])
        colors = cm.jet((pos[:,2] - min_z)/(max_z-min_z))

    pars.set_particles(pos)
    radius = np.ones(num_particles_)*(K**(level-1))*grid_scale
    pars.set_particle_radii(radius)
    # color = np.random.rand(num_particles_, 3).astype(np.float32) * 0.8 + 0.2
    # color = np.zeros((num_particles_, 3)).astype(np.float32)
    pars.set_particle_colors(colors)


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
    color_ = color.to_numpy()
    render_map_to_particles(pars, pos_, color_, num_export_particles[None], level)

    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    # gui.text(content=f'Level {level:.2f} num_particles {num_export_particles[None]} incress =; decress -',
    #         pos=(0, 0.8),
    #         font_size=40,
    #         color=0xffffff)
    gui.text(content=f'Level {level:.2f} grid_scale {(K**(level))*grid_scale} incress =; decress -',
            pos=(0, 0.8),
            font_size=40,
            color=0x0)

    gui.show()
    return level, pos_


if __name__ == '__main__':
    gui = ti.GUI('TaichiOctomap', (RES, RES))

    level = R//2
    scene = tina.Scene(RES)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale*2, theta=-1.0, center=(0, 0, map_scale/2))
    Broot.deactivate_all()

    #Level = 0 most detailed
    random_init_octo(1000)

    while gui.running:
        level, _ = handle_render(scene, gui, pars, level)