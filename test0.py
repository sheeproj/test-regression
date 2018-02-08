import numpy as np
import matplotlib.pyplot as plt

a = 1.0
b = 0.0
lr = 0.0001 # learning rate
loop = 100

def activate(x):
    return np.add(np.dot(a,x), b)

def dactivate():
    return a

def loss(y, yl):
    return (y - yl)*(y - yl)/2.0

def dloss(y, yl):
    return y - yl

# training data sy = sx0 + sx1
sx = np.array([[0,0],[1,2],[1,3],[2,1],[10,10]])
sy = np.array([np.sum(x) for x in sx])

np.random.seed(0)

w0 = 2*np.random.random((2,2)) - 1
w1 = 2*np.random.random((2,1)) - 1

error = np.zeros(loop)
for epoch in range(loop):

    #print('=== epoch{0} ==='.format(epoch))

    dw0 = np.zeros((2,2))
    dw1 = np.zeros((2,1))

    for i in range(len(sx)):

        # forward propagation
        l0 = sx[i]
        l1 = np.dot(w0,l0)
        l2 = activate(l1)
        l3 = np.dot(w1.T,l2) # max

        # error
        error[epoch] += loss(l3, sy[i])
        #print('sample{0} : act {1}, est {2}, error {3}'.format(i, sy[i], l3, error))

        # back-propagation
        delta_error = dloss(l3, sy[i])
        d = delta_error*l2.reshape((2,1))

        dw1 += lr*d
        dw0 += lr*d*(dactivate()*l1)

    #print(dw0, dw1)
    # udpate weight
    w0 = w0 - dw0
    w1 = w1 - dw1

x = np.array([x for x in range(loop)])
y = np.array([y for y in error])

plt.subplot(111)
plt.plot(x, y)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

'''
print('test')
print('w0', w0)
print('w1', w1)
l0 = np.array([2,2])
l1 = np.dot(w0,l0)
l2 = activate(l1)
l3 = np.dot(w1.T,l2) # max
print(l3)
'''
