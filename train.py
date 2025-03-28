import pandas as pd
import numpy as np
from skfeature.function.information_theoretical_based import JMI, DISR
from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import math
from convfxp import ConvFxp
from mlp_fxp_ps import mlp_fxp_ps
from keras.utils import to_categorical
import itertools

import tensorflow
from tensorflow.keras import layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def eval_class(yte, mpred):
    print("new accuracy score: ", accuracy_score(mpred, yte))

def get_maxabs (l):
    return max(max(abs(np.reshape(n, -1))) for n in l)

def get_max (l):
    return max(max(np.reshape(n, -1)) for n in l)

def get_min (l):
    return min(min(np.reshape(n, -1)) for n in l)

def get_width (a):
    b=math.floor(a)
    return b.bit_length()


def convertCoef (mdata, fxp, toFP):
 transp=[np.transpose(i).tolist() for i in mdata]
 coefficients=[]
 for i, l in enumerate(transp):
    newn=list()
    for j,n in enumerate(l):
        neww=list()
        for k,w in enumerate(n):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        newn.append(neww)
    coefficients.append(newn)
 return coefficients

#similar to the above, should be combined
def convertIntercepts (mdata, lfxp, toFP):
 transp=[np.transpose(i).tolist() for i in mdata]
 intercepts=[]
 for i, l in enumerate(transp):
        fxp=lfxp[i]
        neww=list()
        for k,w in enumerate(l):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        intercepts.append(neww)
 return intercepts

def create_cnn(num_filters, kernel_size, activation, neurons, input_shape=(4, 1), num_classes=2, optimizer='adam'):
    model = models.Sequential([
        layers.Conv1D(num_filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape, use_bias=False),  # 4 kernels, kernel size 3
        layers.Flatten(),
        layers.Dense(neurons, activation='relu', use_bias=False),  # Output for num_classes classes
        layers.Dense(num_classes, activation='linear', use_bias=False)  # Output for num_classes classes
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train(dataset, model, method_name):
    if dataset=='spd':
        csv_path="./data/stressPredictData_tabular_win_size_10_clean.csv"
    elif dataset=='wesad':
        csv_path = "./data/90sec_features_n_label_2_SMOTE_True.csv"
        # csv_path = "./WESAD/pop_com_feature_extract_90sec.csv"
    data = pd.read_csv(csv_path, delimiter=',')
    # data=data.dropna(axis=0)

    # Separate features and labels
    y = data['label'].values
    X = data.drop(labels=['label'], axis=1)  # Drop the label column

    # Preprocessing Data
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(num_classes)

    y_categ=to_categorical(y, num_classes=num_classes, dtype='int32')
    if num_classes==2 and (model=="MLP" or model=="CNN"):
        y=np.copy(y_categ)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    folder_path = f'./{model}/{dataset}/{method_name}/float_models'

    # Create directory if it does not exist
    # if not os.path.exists(folder_path):
        # os.makedirs(folder_path)
    # log_file_path = os.path.join(folder_path, "model_log.txt")

    # Delete the log file if it already exists
    # if os.path.exists(log_file_path):
        # os.remove(log_file_path)

    # log_file = open(log_file_path, "a")

    # Apply feature selection
    if method_name=='jmi':
        idx = JMI.jmi(X_train, [np.argmax(y, axis=None, out=None) for y in y_train[:]])
    elif method_name=='disr':
        idx = DISR.disr(X_train, [np.argmax(y, axis=None, out=None) for y in y_train[:]])
    elif method_name=='fisher_score':
        idx = fisher_score.fisher_score(X_train, [np.argmax(y, axis=None, out=None) for y in y_train[:]])
    else:
        print("his feature selection method is not provided yet")
        exit

    feature_sizes = [5, 10, 15, 20, 25, 30]
    np.save(os.path.join(folder_path, f"X_train_{feature_sizes[-1]}.npy"), X_train[:, idx[:feature_sizes[-1]]])
    np.save(os.path.join(folder_path, f"X_test_{feature_sizes[-1]}.npy"), X_test[:, idx[:feature_sizes[-1]]])
    np.save(os.path.join(folder_path, "y_train.npy"), y_train)
    np.save(os.path.join(folder_path, "y_test.npy"), y_test)

    if model=="MLP":
        tuned_parameters = {
            'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (10, 50),(10, 100),(50, 10),(50, 50),(50, 100),(100, 10),(100, 50)],
            'activation': ['relu'],  # activation function
            'solver': ['sgd', 'adam'],  # Solver --> weight optimization - 'sgd', 'adam' (adaptive moment estimation--> better for larger datasets)
            'alpha': [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0],  # L2 penalty (regularization), higher values --> stronger regularization to prevent overfitting
            'learning_rate': ['constant', 'adaptive'],  # learning rate
        }
    elif model=="SVM":
        tuned_parameters = {
            'kernel': ['linear'],  #]
            'gamma': [2e-5, 2**(-4.5), 2e-4, 2**(-3.5), 2e-3, 2**(-2.5), 2e-2, 2**(-1.5), 2e-1, 2**(-0.5), 1, 2**(0.5), 2, 2**(1.5), 4],
            'C': [2e-3, 2**(-2.5), 2e-2, 2**(-1.5), 2e-1, 2**(-0.5), 1, 2**(0.5), 2, 2**(1.5), 4, 2**(2.5), 2e3, 2**(3.5), 2e4]
        }
    elif model=="DT":
        tuned_parameters = dict( 
            criterion = ['gini', 'entropy'],
            splitter = ['best', 'random'],
            min_samples_split = range(2, 10),
            min_samples_leaf = range(1, 10),
            # max_features = ['auto', 'sqrt', 'log2']
    )
    elif model=="CNN":
        tuned_parameters = {
        'num_filters': [4],           # Number of filters in Conv1D
        'kernel_size': [3], 
        'activation': ['relu'],
        # 'neurons' : [16, 32, 64, 128],
        'neurons' : [8],
        # 'activation': ['relu','linear'],
        'batch_size': [64],         # Try different batch sizes
        'epochs': [20, 50],                 # Number of epochs
        'optimizer': ['adam', 'sgd']        # Try different optimizers
    }
        
   

    for num_f in feature_sizes:

        X_train_fs=X_train[:, idx[:num_f]]        
        X_test_fs=X_test[:, idx[:num_f]]     

        if model=="MLP":
            grid = GridSearchCV(MLPClassifier(max_iter=600), tuned_parameters, refit=True, verbose=3, n_jobs=50)
        elif model=="SVM":
            grid = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3, n_jobs=50)
        elif model=="DT":
            grid = GridSearchCV(DecisionTreeClassifier(), tuned_parameters,  refit=True, verbose=3, n_jobs=50)
        elif model=="CNN":
            input_shape = (num_f,1)  # Adjust based on your data shape
            CNN_keras_clf = KerasClassifier(build_fn=create_cnn, input_shape=input_shape, num_classes=num_classes, verbose=0)
            grid = GridSearchCV(CNN_keras_clf,  tuned_parameters,  refit=True, verbose=3, n_jobs=50)

            
        grid.fit(X_train_fs, y_train)

        best_params = grid.best_params_
        accuracy = grid.score(X_test_fs, y_test)
        # log_file.write(f"Feature Size: {num_f}, Accuracy: {accuracy:.4f}, Best Params: {best_params}\n")


        # Save weights and biases for each layer separately
        best_model = grid.best_estimator_
        joblib.dump(best_model, f"{num_f}.joblib")
        quit()

def test(method_name, dataset):

    folder_path = f'./{dataset}/{method_name}'

    feature_sizes = [5, 10, 15, 20, 25, 30]
    X_train=np.load(os.path.join(folder_path, f"X_train_{feature_sizes[-1]}.npy"))
    X_test=np.load(os.path.join(folder_path, f"X_test_{feature_sizes[-1]}.npy"))
    y_train=np.load(os.path.join(folder_path, "y_train.npy"))
    y_test=np.load(os.path.join(folder_path, "y_test.npy"))

    for num_f in feature_sizes:


        model_fs=joblib.load(f"{folder_path}/{num_f}.joblib")
        
        ##floating inference
        X_test_fs=X_test[:,:num_f]
        y_predict=model_fs.predict(X_test_fs)
        print(f"Features: {num_f} Accuracy: {accuracy_score(y_test,y_predict)}")

        ##fixed-point inference

        #4 bit fraction for inputs
        INP_FRAC=4
        #8 bit weights
        W_WIDTH=8
        # trunc 1st relu
        L0trunc=0
    
        inpfxp=ConvFxp(0,0,INP_FRAC)

        w_int=get_width(get_maxabs(model_fs.coefs_))
        w_frac=W_WIDTH-1-w_int
        wfxp=ConvFxp(1,w_int,w_frac)

        bfxp=[]
        trlst=[0]
        for i,c in enumerate(model_fs.intercepts_):
            trlst.append(L0trunc)
            b_int=get_width(get_maxabs(c))
            bfxp.append(ConvFxp(1,b_int,inpfxp.frac+(i+1)*wfxp.frac-trlst[i]))


        # print("Input fxp:", inpfxp.get_fxp())
        # print("Weights fxp:", wfxp.get_fxp())
        # print("Intercepts fxp:", [bfxp[i].get_fxp() for i in range(len(bfxp))])
        # print("Truncate HL:", L0trunc)

        fxp_weights=convertCoef(model_fs.coefs_, wfxp, False)
        fxp_biases=convertIntercepts(model_fs.intercepts_, bfxp, False)

        y_test_fxp=[np.argmax(y, axis=None, out=None) for y in y_test[:]]
        y_train_fxp=[np.argmax(y, axis=None, out=None) for y in y_train[:]]

        fxp_model_fs=mlp_fxp_ps(fxp_weights, fxp_biases, inpfxp, wfxp, L0trunc, y_train_fxp, model_fs)
        # fxp_y_predict=fxp_model_fs.predict(X_test_fs)
        fxp_acc=fxp_model_fs.get_accuracy(X_test_fs,y_test_fxp)

        # print(y_test,fxp_y_predict)
        print(f"Features: {num_f} Accuracy: {fxp_acc}")


if __name__=="__main__":

    # dataset=['wesad','spd']
    dataset=['wesad']
    # dataset=['spd']
    # model=['MLP', 'SVM', 'DT', 'CNN']
    model=['CNN']
    # model=['SVM']
    method_name = ['jmi']

    combinations = list(itertools.product(dataset, model, method_name))

    for combo in combinations:
        train(combo[0], combo[1], combo[2])
    # test(method_name,dataset)