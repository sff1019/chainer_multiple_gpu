import chainer
import chainer.functions as F
import chainer.links as L


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# class ParallelMLP(chainer.Chain):
#     def __init__(self, n_units, n_out, gpu0, gpu1):
#         self.gpu0 = gpu0
#         self.gpu1 = gpu1
#
#
#         super(ParallelMLP, self).__init__()
#         with self.init_scope():
#             self.mlp1_gpu0 = MLP(n_units // 2, n_units).to_gpu(self.gpu0)
#             self.mlp1_gpu1 = MLP(n_units // 2, n_units).to_gpu(self.gpu1)
#
#             self.mlp2_gpu0 = MLP(n_units // 2, n_units).to_gpu(self.gpu0)
#             self.mlp2_gpu1 = MLP(n_units // 2, n_units).to_gpu(self.gpu1)
#
#     def forward(self, x):
#         z0 = self.mlp1_gpu0(x)
#         z1 = self.mlp1_gpu1(F.copy(x, self.gpu1))
#
#         h0 = F.relu(z0 + F.copy(z1, self.gpu0))
#         h1 = F.relu(z1 + F.copy(z0, self.gpu1))
#
#         y0 = self.mlp2_gpu0(h0)
#         y1 = self.mlp2_gpu1(h1)
#
#         y = y0 + F.copy(y1, self.gpu0)
#
#         return y

class ParallelMLP(chainer.Chain):
    def __init__(self, n_units, n_out, gpu0, gpu1, gpu2, gpu3):
        self.gpu0 = gpu0
        self.gpu1 = gpu1
        self.gpu2 = gpu2
        self.gpu3 = gpu3


        super(ParallelMLP, self).__init__()
        with self.init_scope():
            self.mlp1_gpu0 = MLP(n_units // 2, n_units).to_gpu(self.gpu0)
            self.mlp1_gpu1 = MLP(n_units // 2, n_units).to_gpu(self.gpu1)
            self.mlp1_gpu2 = MLP(n_units // 2, n_units).to_gpu(self.gpu2)
            self.mlp1_gpu3 = MLP(n_units // 2, n_units).to_gpu(self.gpu3)

            self.mlp2_gpu0 = MLP(n_units // 2, n_units).to_gpu(self.gpu0)
            self.mlp2_gpu1 = MLP(n_units // 2, n_units).to_gpu(self.gpu1)
            self.mlp2_gpu2 = MLP(n_units // 2, n_units).to_gpu(self.gpu2)
            self.mlp2_gpu3 = MLP(n_units // 2, n_units).to_gpu(self.gpu3)

    def forward(self, x):
        z0 = self.mlp1_gpu0(x)
        z1 = self.mlp1_gpu1(F.copy(x, self.gpu1))
        z2 = self.mlp1_gpu2(F.copy(x, self.gpu2))
        z3 = self.mlp1_gpu3(F.copy(x, self.gpu3))

        h0 = F.relu(z0 + F.copy(z1, self.gpu0) + F.copy(z2, self.gpu0) + F.copy(z3, self.gpu0))
        h1 = F.relu(z1 + F.copy(z0, self.gpu1) + F.copy(z2, self.gpu1) + F.copy(z3, self.gpu1))
        h2 = F.relu(z2 + F.copy(z0, self.gpu2) + F.copy(z1, self.gpu2) + F.copy(z3, self.gpu2))
        h3 = F.relu(z3 + F.copy(z0, self.gpu3) + F.copy(z1, self.gpu3) + F.copy(z2, self.gpu3))

        y0 = self.mlp2_gpu0(h0)
        y1 = self.mlp2_gpu1(h1)
        y2 = self.mlp2_gpu2(h2)
        y3 = self.mlp2_gpu3(h3)

        y = y0 + F.copy(y1, self.gpu0) + F.copy(y2, self.gpu0) + F.copy(y3, self.gpu0)

        return y
