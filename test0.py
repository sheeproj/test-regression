import numpy as np
import matplotlib.pyplot as plt

lr = 0.0001 # learning rate
loop = 1000

def activate_ReLU(x, derivative=False): # ReLU
    condlist = [x < 0, 0 <= x]
    if derivative is False:
        funclist = [lambda x:0, lambda x:x]
    else:
        funclist = [0, 1]
    return np.piecewise(x, condlist, funclist)

def activate_ParametricReLU(x, derivative=False): # Leaky ReLU
    a = 0.9
    condlist = [x < 0, 0 <= x]
    if derivative is False:
        funclist = [lambda x:a*x, lambda x:x]
    else:
        funclist = [a, 1]
    return np.piecewise(x, condlist, funclist)

def loss(y, label, derivative=False):
    if derivative is False:
        return (y - label)*(y - label)/2.0
    else:
        return y - label

if __name__ == "__main__":

    activate = activate_ParametricReLU

    # training data [sy = sx0 + sx1]
    sx = np.array([[0,0],[1,2],[1,3],[2,1],[10,10]])
    sy = np.array([np.sum(x) for x in sx])
    print(sy)

    # initialize weights
    np.random.seed(0)
    w0 = 2*np.random.random((2,2)) - 1
    w1 = 2*np.random.random((2,2)) - 1
    w2 = 2*np.random.random((2,1)) - 1

    error = np.zeros(loop)
    for epoch in range(loop):

        dw0 = np.zeros((2,2))
        dw1 = np.zeros((2,2))
        dw2 = np.zeros((2,1))

        for i in range(len(sx)):

            # forward-propagation
            l1 = np.dot(w0,sx[i])
            l2 = activate(l1)
            l3 = np.dot(w1,l2)
            l4 = activate(l3)
            l5 = np.dot(w2.T,l4) # max

            # calc error
            error[epoch] += loss(l5, sy[i])

            # back-propagation
            delta_error = loss(l5, sy[i], True)

            dw2 += lr*delta_error*l4.reshape((2,1))
            dw1 += lr*delta_error*np.dot(w2.T, activate(l4, True))
            dw0 += lr*delta_error*np.dot(w1, activate(l2, True))

        # udpate weight
        w0 = w0 - dw0
        w1 = w1 - dw1
        w2 = w2 - dw2
        #print('w1', w1)
        #print('w2', w2)

    print('final loss : ', error[loop - 1])

    training_result = []
    for i in range(len(sx)):
        l1 = np.dot(w0,sx[i])
        l2 = activate(l1)
        l3 = np.dot(w1,l2)
        l4 = activate(l3)
        l5 = np.dot(w2.T,l4) # max
        training_result.append(l5[0])
    print('training result : ', training_result)

    x = np.array([x for x in range(loop)])
    y = np.array([y for y in error])

    plt.subplot(111)
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()

