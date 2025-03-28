import sys
from joblib import load
from sklearn.metrics import accuracy_score

import numpy as np
import math
import pandas as pd
from convfxp import ConvFxp
from svm_fxp_ps import svm_fxp_ps
from write_svm_verilog import write_svm_verilog

synth_period=200000.00

def get_maxabs (l):
    return max(max(abs(np.reshape(n, -1))) for n in l)

def get_max (l):
    return max(max(np.reshape(n, -1)) for n in l)

def get_min (l):
    return min(min(np.reshape(n, -1)) for n in l)

def get_width (a):
    b=math.floor(a)
    return b.bit_length()
    #return math.ceil(math.log(a,2))

def convertParamsToFXP (mdata, fxp, toFP):
 convfxp=[]
 if isinstance(mdata[0], np.ndarray) or isinstance(mdata[0], list):
     ldata=mdata
 else:
     ldata=[mdata]
 for i, l in enumerate(ldata):
        neww=list()
        for k,w in enumerate(l):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        convfxp.append(neww)
 if isinstance(mdata[0], np.ndarray) or isinstance(mdata[0], list):
    return convfxp
 else:
    return convfxp[0]

def printData(X, inpfxp, filename):
    f = open(filename, "w")
    for inp in X:
        for w in inp:
            wfx=inpfxp.to_fixed(w)
            f.write(str(wfx)+" ")
        f.write('\n')

dataset_name = sys.argv[1]
float_path = sys.argv[2]
number = sys.argv[3]
mfile = sys.argv[4]
W_WIDTH = int(sys.argv[5])


X_train = np.load(float_path+"/X_train_30.npy")
X_test = np.load(float_path+"/X_test_30.npy")
y_train = np.load(float_path+"/y_train.npy")
y_test = np.load(float_path+"/y_test.npy")

X_train=X_train[:,:int(number.split('_')[0])]
X_test=X_test[:,:int(number.split('_')[0])]

model = load(mfile)
#dataset max size for sim
SIZE = 5000
#4 bit fraction for inputs
INP_FRAC=4
#8 bit weights
# W_WIDTH=8
# W_WIDTH=8

inpfxp=ConvFxp(0,0,INP_FRAC)
#I use the same representation for all the weights --> suboptimal
w_int=min(get_width(get_maxabs(model.coef_)), W_WIDTH-1)
if w_int==W_WIDTH-1:
    print("!An increase in the bitwidth may help the accuracy!")
    print()
w_frac=W_WIDTH-1-w_int
wfxp=ConvFxp(1,w_int,w_frac)

b_int=get_width(get_maxabs(model.intercept_))
bfxp=ConvFxp(1,b_int,inpfxp.frac+wfxp.frac)

print("Input fxp:", inpfxp.get_fxp())
print("Weights fxp:", wfxp.get_fxp())
print("Intercepts fxp:", bfxp.get_fxp())

intercepts=convertParamsToFXP(model.intercept_,bfxp,False)
coefficients=convertParamsToFXP(model.coef_,wfxp,False)

#uncomment to check the fxp values
'''
print("FXP INTERCEPTS:", intercepts)
print("FP INTERCEPTS:",[i.tolist() for i in model.intercept_])
print("FPFXP INTERCEPTS:",convertParamsToFXP(intercepts,bfxp,True))
print("FXP WEIGHTS",coefficients)
print("FP WEIGHTS:",[i.tolist() for i in model.coef_]) 
print("FPFXP WEIGHTS:",convertParamsToFXP(coefficients,wfxp,True))
'''

pred=model.predict(X_test)
isRegr="False"
predname="predo"
numClasses=len(np.unique(y_train))
# numClasses=len(set(y_train))
print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))

# y_test=[np.argmax(y, axis=None, out=None) for y in y_test[:]]
# y_train=[np.argmax(y, axis=None, out=None) for y in y_train[:]]
svmfx=svm_fxp_ps(coefficients,intercepts,inpfxp,wfxp,y_train,model)
svmfx.get_accuracy(X_test,y_test)
print("ACCURACY W/ MLPFX", svmfx.get_accuracy(X_test,y_test))

f=open(f"{dataset_name}exact.v", 'w')
actsizelst,outsize=write_svm_verilog(f,inpfxp.get_width(), wfxp.get_width(), intercepts, coefficients, numClasses)
print("Bitwidth of inputs:", actsizelst )

#write sim datasets
n=min(SIZE,len(X_train))
printData(X_train[0:n], inpfxp, f"{dataset_name}sim.Xtrain")
printData(X_test, inpfxp, f"{dataset_name}sim.Xtest")
f=open(f"{dataset_name}sim.Ytest","w")
np.savetxt(f,y_test,fmt='%d',delimiter='\n')
f.close()
# dump(y_test, f"{dataset_name}sim.Ytest")


## testbench
f=open(f"{dataset_name}top_tb.v","w")
stdoutbckp=sys.stdout
sys.stdout=f
width_a=actsizelst[0]
inp_num=len(X_test[0])
width_o=outsize
print("`timescale 1ns/1ps")
print("`define EOF 32'hFFFF_FFFF")
print("`define NULL 0")
print()
print("module top_tb();")
print()
print(f"    parameter OUTWIDTH={str(width_o)};")
print(f"    parameter NUM_A={str(inp_num)};")
print(f"    parameter WIDTH_A={str(width_a)};")
print()

# localparam period = 200000000.00;
print(f"    localparam period = {synth_period};")
print('''

reg  [WIDTH_A-1:0] at[NUM_A-1:0];
wire [NUM_A*WIDTH_A-1:0] inp;
wire [OUTWIDTH-1:0] out;

wire [WIDTH_A:0] r;

top DUT(.inp(inp),
        .out(out)
        );


integer inFile,outFile,i;
initial
begin
    $display($time, " << Starting the Simulation >>");
        inFile = $fopen("./sim/sim.Xtest","r");
    if (inFile == `NULL) begin
            $display($time, " file not found");
            $finish;
    end
    outFile = $fopen("./sim/output.txt");
    while (!$feof(inFile)) begin
        for (i=0;i<NUM_A;i=i+1) begin
            $fscanf(inFile,"%d ",at[i]);
        end
        $fscanf(inFile,"\\n");
        #(period)
        $fwrite(outFile,"%d\\n", out);
    end
    #(period)
    $display($time, " << Finishing the Simulation >>");
    $fclose(outFile);
    $fclose(inFile);
    $finish;
end


genvar gi;
generate
for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
    assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
end
endgenerate


endmodule''')
f.close()
sys.stdout=stdoutbckp

#write config
uniqYtrain='('+', '.join(str(s) for s in set(y_train) )+ ')'
f=open("circuit_conf.sh", 'w')
conf=[]
conf.append('#!/bin/bash')
conf.append('export PRED_NAME="'+predname+'"')
conf.append('export PRED_NUM='+str(numClasses))
conf.append('#unique values of Ytrain')
conf.append('SIM_YLST='+uniqYtrain)
conf.append('SIM_REGRESSOR="'+str(isRegr)+'";# True or False')
conf.append('#number of fractional bits in the output')
conf.append('SIM_FRACTION_BITS='+str(svmfx.Q))
conf.append('SIM_WIDTH_A="'+str(inpfxp.get_width())+'"')
conf.append('SIM_NUM_A="'+str(len(coefficients[0]))+'"')
conf.append('SIM_OUTWIDTH="'+str(outsize)+'"')
f.write('\n'.join(conf) + '\n')
f.close()

predLst=[]
axLst=[]