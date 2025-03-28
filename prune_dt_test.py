import sys
import numpy as np
from joblib import load


# dataset_name = sys.argv[1]
# float_path = sys.argv[1]
# number = sys.argv[2]
# mfile = sys.argv[3]
float_path = "./DT/wesad/disr/float_models"
number = "5"
mfile = "./DT/wesad/disr/float_models/5.joblib"

X_train = np.load(float_path+"/X_train_30.npy")
X_test = np.load(float_path+"/X_test_30.npy")
y_train = np.load(float_path+"/y_train.npy")
y_test = np.load(float_path+"/y_test.npy")

X_train=X_train[:,:int(number.split('_')[0])]
X_test=X_test[:,:int(number.split('_')[0])]

# print(mfile)
dt_model = load(mfile)

path = dt_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(ccp_alphas)
print(impurities)