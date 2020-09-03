from os import listdir
from os.path import isfile, join
import numpy as np

class DataLoader():
    
    def __init__(self):
        pass

    def get_npz_x_y(self,myfile):
        with np.load(myfile) as npz:
            return npz['x'], npz['y']

    def get_x_y_list(self,file_list):
        X, y = [], []
        for i in range(file_list.shape[0]):
            _X, _y = self.get_npz_x_y(file_list[i])
            X.append(_X)
            y.append(_y)
        return np.array(X), np.array(y)
    
    def get_seq_x_y(self,X,y,len_seq=25):
        assert len(X) == len(y)

        X_seq, y_seq = [], []
        for patient in range(X.shape[0]):
            for s in range(0,len(X[patient]),len_seq):
                e = s+len_seq
                if e > len(X[patient]):
                    break
                X_seq.append(X[patient][s:e]) 
                y_seq.append(y[patient][s:e])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        return X_seq, y_seq
    
    def flatten_x_y(self,X,y,step): 
        assert len(X) == len(y)
        
        _X, _y = [], []
        for i in range(X.shape[0]):
            for j in range(0,len(X[i]),step):
                s = j
                e = j+step
                if e > len(X[i]):
                    break
                _X.append(X[i][s:e])
                _y.append(y[i][s:e])
        return np.array(_X), np.array(_y)


    def __call__(self,mypath,seed=False,step=2,len_seq=25,return_sequences=False):
        assert seed is not False

        files = np.array(sorted([join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f))]))

        idx = np.arange(len(files))
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_files = files[idx[:int(len(idx)*.5)]]
        valid_files = files[idx[int(len(idx)*.5):int(len(idx)*.75)]]
        test_files = files[idx[int(len(idx)*.75):]]

        X_train, y_train = self.get_x_y_list(train_files)
        X_valid, y_valid = self.get_x_y_list(valid_files)
        X_test, y_test = self.get_x_y_list(test_files)
        
        if return_sequences:
            X_seq_train, y_seq_train = self.get_seq_x_y(X_train, y_train,len_seq=len_seq)
            X_seq_valid, y_seq_valid = self.get_seq_x_y(X_valid, y_valid,len_seq=len_seq)
            X_seq_test, y_seq_test = self.get_seq_x_y(X_test,y_test,len_seq=len_seq)
            
            X_train, y_train = self.flatten_x_y(X_seq_train, y_seq_train,1)
            X_valid, y_valid = self.flatten_x_y(X_seq_valid, y_seq_valid,1)
            X_test, y_test = self.flatten_x_y(X_seq_test, y_seq_test,1)
            
            return X_train, y_train, X_valid, y_valid, X_test, y_test, X_seq_train, y_seq_train, X_seq_valid, y_seq_valid, X_seq_test, y_seq_test
        else:
            X_train, y_train = self.flatten_x_y(X_train, y_train,step)
            X_valid, y_valid = self.flatten_x_y(X_valid, y_valid,step)
            X_test, y_test = self.flatten_x_y(X_test, y_test,step)
            return X_train, y_train, X_valid, y_valid, X_test, y_test