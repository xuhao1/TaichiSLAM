# This file is an easy octomap implentation based on taichi lang
# The propose of this file is to explore the features of the taichi lang.
#
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
ti.init(arch=ti.cpu)

RES = 1024
K = 2
R = 7
N = K**R

Broot = ti.root
B = ti.root
for r in range(R):
    B = B.bitmasked(ti.ijk, (K, K, K))

qt = ti.field(ti.f32)
B.place(qt)

img = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

print('The quad tree layout is:\n', qt.snode)