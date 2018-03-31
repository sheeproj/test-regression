# test0_nnabla.py

This script predicts an output value of simple linear function using the NNabla solver.
The objective function is

```math
y = f(x0, x1) = x0 + x1
```

This is just a test-code for my practice.

# test1_nnabla.py

This script predicts an output value of simple linear function using the NNabla solver.
The objective function is

```math
y = f(x0, x1) = sin(x0) + cos(x1)
```

With the default configuration, the network layer size is defined as [8 -> 8] and batch_size is 100.
It can show how network layer size and batch_size affect prediction accuracy.
Here are some results.

### network layer = [2 -> 2], batch_size = 100

![Alt text](img/2-2-100-loss.png?raw=true "Title")
![Alt text](img/2-2-100-pred.png?raw=true "Title")

### network layer = [4 -> 4], batch_size = 100

![Alt text](img/4-4-100-loss.png?raw=true "Title")
![Alt text](img/4-4-100-pred.png?raw=true "Title")

### network layer = [8 -> 8], batch_size = 100

![Alt text](img/8-8-100-loss.png?raw=true "Title")
![Alt text](img/8-8-100-pred.png?raw=true "Title")

### network layer = [8 -> 8], batch_size = 500

![Alt text](img/8-8-500-loss.png?raw=true "Title")
![Alt text](img/8-8-500-pred.png?raw=true "Title")
