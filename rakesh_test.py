import sys
import numpy as np
from joblib import load


# dataset_name = sys.argv[1]
# float_path = sys.argv[1]
# number = sys.argv[2]
# mfile = sys.argv[3]
float_path = "pruned_joblibs/jmi"
number = "5"
mfile = "CNN_changes/pruned_joblibs/jmi/5/5_sparsity_0_2.joblib"

X_train = np.load(float_path+"/X_train_30.npy")
X_test = np.load(float_path+"/X_test_30.npy")
y_train = np.load(float_path+"/y_train.npy")
y_test = np.load(float_path+"/y_test.npy")

X_train=X_train[:,:int(number.split('_')[0])]
X_test=X_test[:,:int(number.split('_')[0])]

# print(mfile)
loaded_model = load(mfile)
model = loaded_model.model