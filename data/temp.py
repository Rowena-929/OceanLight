import numpy as np

data = np.load('./climatology/mean.npy')
new = data[:,:,::3,::3]
np.save('./climatology/mean.npy', new)