import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla import solvers as S

# data
def random_data_provider(n):
    x = np.random.uniform(-math.pi, math.pi, size=(n, 2))
    y = np.array([math.sin(a[0]) + math.cos(a[1]) for a in x]).reshape(n,1)
    #y = np.array([math.sin(a[0])*math.cos(a[1]) for a in x]).reshape(n,1)
    return x, y

# start NNabla
nn.clear_parameters()
batchsize = 100
xdim = 2
nlayers = [8,8]
num_iter = 10000

x = nn.Variable([batchsize, xdim], need_grad=True)
label = nn.Variable([batchsize, 1], need_grad=True)

# create network
h = x
for i, l in enumerate(nlayers):
    h = F.relu(PF.affine(h, l, name='fc{0}'.format(i)))
y = PF.affine(h, 1, name='fc')

loss = F.reduce_mean(F.squared_error(y, label))

solver = S.Adam(alpha=0.01)
solver.set_parameters(nn.get_parameters())

for name, param in nn.get_parameters().items():
    print(name, param)

# training
training_loss = []
for i in range(num_iter):

    # Sample data and set them to input variables of training.
    xx, ll = random_data_provider(batchsize)
    x.d = xx
    label.d = ll

    # Forward propagation given inputs.
    loss.forward(clear_no_need_grad=True)

    # Parameter gradients initialization and gradients computation by backprop.
    solver.zero_grad()
    loss.backward(clear_buffer=True)

    # Apply weight decay and update by Adam rule.
    solver.weight_decay(1e-6)
    solver.update()

    # Just print progress.
    if i % (num_iter/10) == 0 or i == num_iter - 1:
        print("Loss@{:4d}: {}".format(i, loss.d))

    training_loss.append(np.copy(loss.d))

# draw loss graph
plt.subplot(111)
plt.plot(np.array([i for i in range(len(training_loss))]), np.array(training_loss))
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

# test
tx, ty = random_data_provider(batchsize)
results = []
for i in range(0, tx.shape[0], batchsize):
    x.d = tx[i:i+batchsize]
    y.forward()
    results.append(np.copy(y.d))
results = np.array(results)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(len(tx)):
    xs = np.hstack((xs,tx[i][0]))
    ys = np.hstack((ys,tx[i][1]))
    zs = np.hstack((zs,ty[i]))

ax.scatter(xs, ys, zs, color='green', marker='o', label='ground-truth')
ax.scatter(xs, ys, results, color='red', marker='o', label='predicted')
ax.legend()
plt.show()
