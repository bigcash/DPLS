import numpy as np
import os
import random

if __name__ == '__main__':
    random.seed(5000)
    data_dir = '/path/to/fineweb-edu100B'
    train_file_list = list(filter(lambda x: x.endswith('.bin') and x.startswith('fineweb_train'), os.listdir(data_dir)))
    train_data_list = [np.memmap(os.path.join(data_dir, file), dtype=np.uint16, mode='r') for file in train_file_list]
    for i in range(len(train_file_list)):
        data = train_data_list[i]
        indices = np.where(data == 50256)[0]
        for j in range(1, len(indices), 1):
            every = data[indices[j - 1] + 1:indices[j]+1]
            tokens = list(every)
            print(' '.join([str(t) for t in tokens]))
        # last = data[indices[-1] + 1:]

