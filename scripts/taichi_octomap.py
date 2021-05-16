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
R = 12
N = K**R
map_scale = 100
grid_scale = map_scale/N
max_num_particles = 10000

Broot = ti.root
B = ti.root
for r in range(R):
    B = B.pointer(ti.ijk, (K, K, K))
    # B = B.bitmasked(ti.ijk, (K, K, K))

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)


#qt.parent is the deepest of bitmasked
qt = ti.field(ti.f32)
B.place(qt)

print(f'The map voxel is:[{N}x{N}x{N}] map scale:[{map_scale}mx{map_scale}mx{map_scale}m], grid scale {grid_scale:3.3f}m')


@ti.kernel
def random_init_octo():
    for i in range(30):
        x_ = ti.random(dtype = int)%N
        y_ = ti.random(dtype = int)%N
        z_ = ti.random(dtype = int)%N
        qt[x_, y_, z_] = 1

@ti.kernel
def get_voxel_to_particles(level: ti.template()):
    # Number for level
    n = K**(R-level)
    level_grid_scale = 2**level
    num_particles[None] = 0
    for _i, _j, _k in ti.ndrange(n, n, n):
        i = level_grid_scale*_i
        j = level_grid_scale*_j
        k = level_grid_scale*_k
        if ti.is_active(qt.parent(level+1), [i, j, k]):
            index = ti.atomic_add(num_particles[None], 1)
            x[index][0] = i*grid_scale
            x[index][1] = j*grid_scale
            x[index][2] = k*grid_scale
                


def render_map_to_particles(pars, pos_, num_particles_, level):
    #print(f"set_particles {num_particles_}")
    pos = pos_[0:num_particles_,:]
    pars.set_particles(pos)
    radius = np.ones(num_particles_)*(2**(level))*grid_scale
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
    scene.init_control(gui, radius=map_scale*2, center=(map_scale/2, map_scale/2, map_scale/2))
    Broot.deactivate_all()

    random_init_octo()

    #Level = 0 most detailed
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
        print("level", level, "num_particles ", num_particles[None])
        get_voxel_to_particles(level)
        pos_ = x.to_numpy()
        render_map_to_particles(pars, pos_, num_particles[None], level)

        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.text(content=f'Level {level:.2f} num_particles {num_particles[None]} incress =; decress -',
                pos=(0, 0.8),
                color=0xffffff)
        gui.show()
