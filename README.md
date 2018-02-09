# test1_nnabla.py

This predicts a simple linear function using the NNabla solver.

```math
y = f(x0, x1) = sin(x0) + cos(x1)
```

The size of network layer is defined as [8 -> 8] and default batch_size is 100.
This is useful for testing how network layer and batch_size affect prediction accuracy.
Here are some results.

- network layer = [2 -> 2], batch_size = 100

- network layer = [4 -> 4], batch_size = 100

- network layer = [8 -> 8], batch_size = 100

- network layer = [8 -> 8], batch_size = 500
