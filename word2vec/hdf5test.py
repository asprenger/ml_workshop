import h5py
import numpy as np

filename = 'file.hdf5'
example_size = 10

# Write data to HDF5
with h5py.File(filename, 'w') as data_file:

    x_train_ds = None
    y_train_ds = None
    for i in range(10000000):
        if i%10000==0:
            print(i)
        x_train = np.empty((1,example_size))
        x_train.fill(i)
        y_train = np.array([i])

        if x_train_ds == None:
            x_train_ds = data_file.create_dataset('x_train', 
                                                  data=x_train, 
                                                  chunks=(100,example_size), 
                                                  maxshape=(None,example_size))
            y_train_ds = data_file.create_dataset('y_train', 
                                          data=y_train, 
                                          chunks=(100,), 
                                          maxshape=(None,))

        else:    
            x_train_ds.resize(x_train_ds.shape[0]+1, axis=0)
            x_train_ds[i,:] = x_train

            y_train_ds.resize(y_train_ds.shape[0]+1, axis=0)
            y_train_ds[i] = y_train

print('Done writing')

f = h5py.File(filename, 'r')
x_train = f['x_train'].value
print(x_train.shape)
y_train = f['y_train'].value
print(y_train.shape)