# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import tina
import time

from tina.util.matrix import lookat

ti.init(arch=ti.cpu, debug=True)

RES = 1024
K = 2
R = 11
Rz = 8
N = K**R
Nz = K**Rz
map_scale = 100
map_scale_z = 10
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

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)


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
    num_particles[None] = 0

    for i, j, k in qt.parent(level+1):
        index = ti.atomic_add(num_particles[None], 1)
        x[index][0] = i*grid_scale
        x[index][1] = j*grid_scale
        x[index][2] = k*grid_scale


def render_map_to_particles(pars, pos_, num_particles_, level):
    #print(f"set_particles {num_particles_}")
    pos = pos_[0:num_particles_,:]
    pars.set_particles(pos)
    radius = np.ones(num_particles_)*(K**(level))*grid_scale
    pars.set_particle_radii(radius)
    # color = np.random.rand(num_particles_, 3).astype(np.float32) * 0.8 + 0.2
    color = np.ones((num_particles_, 3)).astype(np.float32)
    pars.set_particle_colors(color)

if __name__ == '__main__':
    gui = ti.GUI('TaichiOctomap', (2048, 2048))
    level = R//2
    scene = tina.Scene(2048)
    pars = tina.SimpleParticles()
    material = tina.Classic()
    scene.add_object(pars, material)
    scene.init_control(gui, radius=map_scale*2, theta=-1.0, center=(map_scale/2, map_scale/2, map_scale/2))
    Broot.deactivate_all()


    #Level = 0 most detailed
    random_init_octo(1000)

    while gui.running:

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
        render_map_to_particles(pars, pos_, num_particles[None], level)

        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.text(content=f'Level {level:.2f} num_particles {num_particles[None]} incress =; decress -',
                pos=(0, 0.8),
                font_size=40,
                color=0xffffff)
        gui.show()
