import nnabla as nn  # Abbreviate as nn for convenience.
import numpy as np
import matplotlib.pyplot as plt

import nnabla.functions as F
import nnabla.parametric_functions as PF

nn.clear_parameters()
batchsize = 8

x = nn.Variable([batchsize, 2])
print(x.data)
