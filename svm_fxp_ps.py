import numpy as np
from sklearn.metrics import accuracy_score
from convfxp import to_float

class svm_fxp_ps:
    
 def __init__(self, coefs, intercept, inpfxp, wfxp, y_train, model):
     self.coefs = coefs
     self.intercept = intercept
     self.inpfxp=inpfxp
     self.Q=inpfxp.frac+wfxp.frac
     self.YMAX=max(y_train)
     self.YMIN=min(y_train)
     self.YMAP=list(set(y_train))
     if "SVR" in str(type(model)):
         self.regressor=True
     else:
         self.regressor=False
 def vsign (self, a):
     if a>=0:
         return 0
     else:
         return 1

 def decisionFunction(self,pred):
    ymap=self.YMAP
    dMatrix=np.zeros((len(ymap),len(ymap)))
    ind=0
    for m in range(len(ymap)):
        for n in range(len(ymap)):
            if m<n:
                dMatrix[m][n]=(1-self.vsign(pred[ind]))
                dMatrix[n][m]=self.vsign(pred[ind])
                ind=ind+1
    s=np.sum(dMatrix, axis=1)
    return ymap[np.argmax(s)]

 def predict(self,X_test):
    prediction=[]
    for i, l in enumerate(X_test):
        newl=[]
        for k,w in enumerate(l):
            newl.append(self.inpfxp.to_fixed(w))
        prediction.append( self.predict_one(newl))
    return np.asarray(prediction)

 def get_accuracy(self, X_test, y_test):
    pred=self.predict(X_test)+self.YMIN
    # pred=self.predict(X_test)
    return accuracy_score(pred, y_test)

 def predict_one(self, x):
    inp=x
    layer=0
    out=[]
    for i,neuron in enumerate(self.coefs):
            temp=self.intercept[i]
            for j in range(len(neuron)):
                temp+=neuron[j]*inp[j]
            out.append(temp)
    if self.regressor:
        p=to_float(out[0],self.Q)
        return np.clip(np.round(p),self.YMIN,self.YMAX)
    else:
        # print(out)
        if len(out)!=1:
            p=self.decisionFunction(out)
        else:
            if out[0]>=0:
                p=1
            else:
                p=0
        return p
