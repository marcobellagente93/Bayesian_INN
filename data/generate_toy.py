import numpy as np
import pandas as pd

x = np.random.uniform(0,1, 200000)
y = np.random.uniform(0,1, 200000)

weight_linear    = x
weight_quadratic = x**2

mask_linear    = y < weight_linear
mask_quadratic = y < weight_quadratic

x_linear    = x[mask_linear][:50000]
x_quadratic = x[mask_quadratic][:50000]

y = np.random.uniform(0,1, [50000,1])

linear    = np.concatenate((x_linear.reshape(50000,1), y), -1)
quadratic = np.concatenate((x_quadratic.reshape(50000,1), y), -1)

s1 = pd.HDFStore('2d_linear.h5')
s2 = pd.HDFStore('2d_quadratic.h5')
s1.append('data', pd.DataFrame(linear))
s2.append('data', pd.DataFrame(quadratic))
s1.close()
s2.close()
