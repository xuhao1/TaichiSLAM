import taichi as ti
import numpy as np

MAX_PARAM_DIM = 4

#This file stands a genernal purpose Non-linear-least-square solver written with taichi.
class CostFunction:
    def __init__(self):
        self.indices = []
        pass

    @ti.func
    def evaluate(self, field):
        pass

    def set_indices(self, indices):
        self.indices = ti.field(ti.i32, shape=(len(indices),2))
        self.indices.from_numpy(indices)
    
    def residual_dim(self):
        return 0

@ti.data_oriented
class TaichiNNLS:
    def __init__(self, verbose=False):
        self.param_count = 0
        self.params = {}
        self.params_index_field = {}
        self.cost_function_pairs = []
        self.params_field = None
        self.local_param = {}
        self.verbose = verbose
        self.loss_field = None
        self.size_residual = 0
        self.size_param = 0
    
    def pre_solve(self):
        size_param = 0
        for param_id in self.params:
            param = self.params[param_id] #param numpy array
            size_param += len(param)
        if self.verbose:
            print(f"[TaichiNNLS] param size {size_param}")
        self.params_field = ti.field(ti.f32, shape=size_param, needs_grad=True)
        params_numpy = np.zeros(shape=size_param)
        index_param = 0
        for param_id in self.params:
            param = self.params[param_id] #param numpy array
            self.params_index_field[param_id] = [index_param, len(param)]
            params_numpy[index_param:index_param + len(param)] = param
            if self.verbose:
                print(f"[TaichiNNLS] param {id(param)} index {index_param}")
            index_param += len(param)
        self.params_field.from_numpy(params_numpy)
        self.size_param = size_param

        size_residual = 0
        for i in range(len(self.cost_function_pairs)):
            cost_func_pair = self.cost_function_pairs[i]
            cost_func = cost_func_pair[0]
            params = cost_func_pair[1]
            indices = [] 
            for i in range(len(params)):
                _id = id(params[i])
                indices.append(self.params_index_field[_id])
            indices = np.array(indices)
            cost_func.set_indices(indices)
            size_residual += cost_func.residual_dim()
            print("[TaichiNNLS] cost function", cost_func, "indices", indices, "residual size", cost_func.residual_dim())
        print("Total residuals ", size_residual)
        self.loss_field = ti.field(ti.f32, shape=size_residual, needs_grad=True)
        self.size_residual = size_residual

    def add_cost_function(self, cost_func, *params):
        self.cost_function_pairs.append((cost_func, params))
        for param in params:
            if id(param) not in self.params:
                self.params[id(param)] = param
                if self.verbose:
                    print(f"[TaichiNNLS] adding param {param} with id {id(param)}")
    
    @ti.kernel
    def evaluate_test_kernel(self, cost_func: ti.template()):
        cost_func.evaluate(self.params_field)

    def evaluate_test(self):
        print(self.loss_field)
        for i in range(self.size_residual):
            self.loss_field.grad[i] = 1
        for i in range(len(self.cost_function_pairs)):
            cost_func_pair = self.cost_function_pairs[i]
            cost_func = cost_func_pair[0]
            self.evaluate_test_kernel(cost_func)
            self.evaluate_test_kernel.grad(cost_func)
        for i in range(self.size_param):
            print(self.params_field.grad[i])