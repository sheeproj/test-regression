import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla import solvers as S

sx = np.array([[0,0],[1,2],[1,3],[2,1],[10,10]])
sy = np.array([np.sum(x) for x in sx])

# start NNabla
nn.clear_parameters()
batchsize = len(sx)

x = nn.Variable.from_numpy_array(sx, need_grad=True)

label = nn.Variable.from_numpy_array(sy)
label = label.reshape((batchsize, 1))

# create network
with nn.parameter_scope("fc1"):
    h0 = F.relu(PF.affine(x, 2))
with nn.parameter_scope("fc2"):
    h1 = F.relu(PF.affine(h0, 2))
y = PF.affine(h1, 1, name='fc')

loss = F.reduce_mean(F.squared_error(y, label))

solver = S.Adam(alpha=0.01)
solver.set_parameters(nn.get_parameters())

def random_data_provider(n):
    x = np.random.uniform(0, 10, size=(n, 2))
    y = np.array([np.sum(a) for a in x]).reshape(n,1)
    return x, y

# training
num_iter = 10000
for i in range(num_iter):

    # Sample data and set them to input variables of training.
    xx, ll = random_data_provider(batchsize)
    x.d = sx
    #label.d = ll
    # Forward propagation given inputs.
    loss.forward(clear_no_need_grad=True)
    # Parameter gradients initialization and gradients computation by backprop.
    solver.zero_grad()
    loss.backward(clear_buffer=True)
    # Apply weight decay and update by Adam rule.
    solver.weight_decay(1e-6)
    solver.update()
    # Just print progress.
    if i % 100 == 0 or i == num_iter - 1:
        print("Loss@{:4d}: {}".format(i, loss.d))

# validation
x.d = sx
y.forward()
for i in range(len(y.d)):
    print('label<->act : {0}<->{1}'.format(label.d[i], y.d[i]))
