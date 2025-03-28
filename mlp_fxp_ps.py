import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from convfxp import ConvFxp, to_float

class mlp_fxp_ps:
    
 def __init__(self, coefs, intercept, inpfxp, wfxp, L0trunc, y_train, model):
     self.coefs = coefs
     self.intercept = intercept
     self.inpfxp=inpfxp
     if len(intercept)==2:
        self.trunclst=[L0trunc,0]
     elif len(intercept)==3:
        self.trunclst=[L0trunc,L0trunc,0]
     self.Q=inpfxp.frac+2*wfxp.frac-L0trunc
     self.YMAX=max(y_train)
     self.YMIN=min(y_train)
     if "MLPRegressor" in str(type(model)):
         self.regressor=True
     else:
         self.regressor=False

 def predict(self,X_test):
    prediction=[]
    for i, l in enumerate(X_test):
        newl=[]
        for k,w in enumerate(l):
            newl.append(self.inpfxp.to_fixed(w))
        prediction.append( self.predict_one(newl))
    return np.asarray(prediction)

 def get_accuracy(self, X_test, y_test):
    pred=self.predict(X_test)
    if self.regressor:
        pred=np.clip(np.round(pred),self.YMIN,self.YMAX)
    else:
        pred=pred+self.YMIN
    return accuracy_score(pred, y_test)

 def predict_one(self, x):
    inp=x
    layer=0
    out=[]
    for layer in range(len(self.coefs)):
        for i,neuron in enumerate(self.coefs[layer]):
            temp=self.intercept[layer][i]
            for j in range(len(neuron)):
                temp+=neuron[j]*inp[j]
            if (temp<0):
                temp = 0
            temp=temp>>self.trunclst[layer]
            out.append(temp)
        inp=list(out)
        out=[]
    if self.regressor:
        return to_float(inp[0],self.Q)
    else:
        return np.argmax(inp)
