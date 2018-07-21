def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def read_idx(filename):
    import struct
    import numpy as np
    """
    Author: Tyler Neylon
    https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
    
def load_mnist_idx_from_folder(folder, pattern = '*ubyte'):
    """
        Reads Fashion-MNIST data from folder.
        
        Returns a dictionary of training and test data.
    """
    import os
    import fnmatch
    # crawl directory and grab filenames
    names = []
    for path, subdirs, files in os.walk(folder):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                names.append(os.path.join(path, filename))
                
    num_files = len(names)
    for i in range(len(names)):
        print("{}".format(names[i]))
        
    print("\nThere are {} files.".format(num_files))
    
    # read the files into a numpy array
    data = {}
    for i in range(num_files):
        if 'train' in names[i]:
            if 'images' in names[i]:
                data['train_imgs'] = read_idx(names[i]) 
            else:
                data['train_labels'] = read_idx(names[i])
        else:
            if 'images' in names[i]:
                data['test_imgs'] = read_idx(names[i])
            else:
                data['test_labels'] = read_idx(names[i])
    return data