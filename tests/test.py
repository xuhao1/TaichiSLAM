import taichi as ti
ti.init(arch=ti.cpu, dynamic_index=True)

@ti.kernel
def test_kernel():
    vertlist = ti.Matrix([[0., 0., 0.] for i in range(3)])
    p = ti.Vector([1, 2, 3])
    vertlist[1, :] = p.transpose()
    print("vertlist assign by slicing", vertlist)

    vertlist[1, 0] = p[0]
    vertlist[1, 1] = p[1]
    vertlist[1, 2] = p[2]

    print("assign by indexing", vertlist)

@ti.kernel
def test_kernel2():
    p0 = ti.Vector([1, 2, 3], ti.i32)
    p1 = ti.Vector([8, 6, 4], ti.i32)
    print(p0)
    mu = 0.5
    p = p0
    p = p0 + mu*(p1-p0)
    print("interlop", p)
    p = ti.Vector([1.4, 2.4, 3.2], ti.f32)
    print("reassign to float", p)

    a = 1
    print(a)
    a = 1.5
    print("1.5", a)


params_field = ti.field(ti.f32, shape=12)

@ti.kernel
def init_vec_by_field(field: ti.template(), a:ti.template()):
    v = ti.Vector([0. for i in range(10)])
    for i in range(a):
        v[i] = field[i]

# test_kernel()
# test_kernel2()
init_vec_by_field(params_field, 5)
