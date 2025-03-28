import sys
# from sklearn.model_selection import train_test_split
from joblib import load
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import accuracy_score
# import torch

import numpy as np
import math
import pandas as pd
from convfxp import ConvFxp
from mlp_fxp_ps import mlp_fxp_ps
# from write_mlp_verilog import write_mlp_verilog
from write_mlp_verilogFA import write_mlp_verilog


synth_period=200000.00

def eval_reg(ytr, yte, mpred):
    new_preds=[]
    for reg_pred in mpred:
        new_preds.append(min(max(ytr),max(min(ytr),round(reg_pred))))
    print("new accuracy score: ", accuracy_score(new_preds, yte))

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
    #return math.ceil(math.log(a,2))


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

# print(mfile)
model = load(mfile)
#dataset max size for sim
SIZE = 5000
#4 bit fraction for inputs
INP_FRAC=4
#8 bit weights
# W_WIDTH=8
# trunc 1st relu
L0trunc=0
# trlst=[0,L0trunc]
# trlst=[0,L0trunc,L0trunc]

inpfxp=ConvFxp(0,0,INP_FRAC)

#I use the same representation for all the weights --> suboptimal
w_int=get_width(get_maxabs(model.coefs_))
w_frac=W_WIDTH-1-w_int
wfxp=ConvFxp(1,w_int,w_frac)

bfxp=[]
trlst=[0]
for i,c in enumerate(model.intercepts_):
    trlst.append(L0trunc)
    b_int=get_width(get_maxabs(c))
    bfxp.append(ConvFxp(1,b_int,inpfxp.frac+(i+1)*wfxp.frac-trlst[i]))


print("Input fxp:", inpfxp.get_fxp())
print("Weights fxp:", wfxp.get_fxp())
print("Intercepts fxp:", [bfxp[i].get_fxp() for i in range(len(bfxp))])
print("Truncate HL:", L0trunc)

intercepts=convertIntercepts(model.intercepts_,bfxp,False)
coefficients=convertCoef(model.coefs_,wfxp,False)

#uncomment to check the fxp values
'''
print("FXP INTERCEPTS (TRANSP):", intercepts)
print("FP INTERCEPTS:",[i.tolist() for i in model.intercepts_])
print("FPFXP INTERCEPTS:",convertIntercepts(intercepts,bfxp,True))
print("FXP WEIGHTS (TRANP)",coefficients)
print("FP WEIGHTS:",[i.tolist() for i in model.coefs_]) 
print("FPFXP WEIGHTS:",convertCoef(coefficients,wfxp,True))
'''

# y_test = Y_test
# y_train = Y_train

pred=model.predict(X_test)
isRegr="False"
predname="predo"
if "MLPRegressor" in str(type(model)):
        predname="out"
        isRegr="True"
        pred=np.clip(np.round(pred),min(y_test),max(y_test))
print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))

y_test=[np.argmax(y, axis=None, out=None) for y in y_test[:]]
y_train=[np.argmax(y, axis=None, out=None) for y in y_train[:]]
mlpfx=mlp_fxp_ps(coefficients,intercepts,inpfxp,wfxp,L0trunc,y_train,model)
mlpfx.get_accuracy(X_test,y_test)
print("ACCURACY W/ MLPFX", mlpfx.get_accuracy(X_test,y_test))

f=open(f"{dataset_name}exact.v", 'w')
actsizelst,outsize=write_mlp_verilog(f,inpfxp.get_width(), L0trunc, wfxp.get_width(), intercepts, coefficients)
print("Bitwidth of activations:", actsizelst[:-1] )

#write sim datasets
n=min(SIZE,len(X_train))
printData(X_train[0:n], inpfxp, f"{dataset_name}sim.Xtrain")
printData(X_test, inpfxp, f"{dataset_name}sim.Xtest")
f=open(f"{dataset_name}sim.Ytest","w")
np.savetxt(f,np.array(y_test).astype(int),fmt='%d',delimiter='\n')
# np.savetxt(f,Y_test.astype(int),fmt='%d',delimiter='\n')
f.close()
# dump(y_test, "sim.Ytest")

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

# localparam period = 250000000.00;
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
conf.append('export PRED_NUM='+str(len(coefficients[-1])))
conf.append('#unique values of Ytrain')
conf.append('SIM_YLST='+uniqYtrain)
conf.append('SIM_REGRESSOR="'+isRegr+'";# True or False')
conf.append('#number of fractional bits in the output')
conf.append('SIM_FRACTION_BITS='+str(mlpfx.Q))
conf.append('SIM_WIDTH_A="'+str(inpfxp.get_width())+'"')
conf.append('SIM_NUM_A="'+str(len(coefficients[0][0]))+'"')
conf.append('SIM_OUTWIDTH="'+str(outsize)+'"')
f.write('\n'.join(conf) + '\n')
f.close()

# predLst=[]
# axLst=[]

# #approximation AX0
# area_AC = 0
# area_AX = 0
# ax_coeff=[]
# ax=[4, 4]
# print("ax params:", ax)

# tempax=list(ax)
# for i,layer in enumerate(coefficients):
#     newLayer=[]
#     for neuron in layer:
#         newNeuron=[]
#         for w in neuron:
#             wa=mult_area(actsizelst[i])
#             area_AC+=wa.area[w]
#             wax=wa.wamin_sgn(w,ax[i])
#             area_AX+=wa.area[wax]
#             ax[i]=ax[i]*-1
#             newNeuron.append(wax)
#         newLayer.append(newNeuron)
#     ax_coeff.append(newLayer)
# #print ("ax_coef",ax_coeff)
# print ("ax0:", area_AC, area_AX, area_AX/area_AC)

# mlpfx_ax=mlp_fxp_ps(ax_coeff,intercepts,inpfxp,wfxp,L0trunc,y_train,model)
# pred=mlpfx_ax.get_accuracy(X_test,y_test)
# predLst.append(pred)
# axLst.append(mlpfx_ax)
# print("ACCURACY W/ MLPFX AX0", pred)


# ax=tempax
# #approximation AX1
# area_AC = 0
# area_AX = 0
# ax_coeff=[]
# for i,layer in enumerate(coefficients):
#     newLayer=[]
#     for neuron in layer:
#         wa=mult_area(actsizelst[i])
#         approx=ax_weight(neuron,ax[i], actsizelst[i])
#         newNeuron=approx.find_ax()
#         area_AC+=sum(wa.area[w] for w in neuron)
#         area_AX+=sum(wa.area[w] for w in newNeuron)
#         newLayer.append(newNeuron)
#     ax_coeff.append(newLayer)
# #print ("ax_coef",ax_coeff)
# print ("ax1:", area_AC, area_AX, area_AX/area_AC)

# mlpfx_ax=mlp_fxp_ps(ax_coeff,intercepts,inpfxp,wfxp,L0trunc,y_train,model)
# pred=mlpfx_ax.get_accuracy(X_test,y_test)
# predLst.append(pred)
# axLst.append(mlpfx_ax)
# print("ACCURACY W/ MLPFX AX1", pred)

# #approximation AX2
# area_AC = 0
# area_AX = 0
# ax_coeff=[]
# for i,layer in enumerate(coefficients):
#     newLayer=[]
#     for neuron in layer:
#         wa=mult_area(actsizelst[i])
#         approx=ax_weight_all(neuron,ax[i], actsizelst[i])
#         newNeuron=approx.find_ax()
#         area_AC+=sum(wa.area[w] for w in neuron)
#         area_AX+=sum(wa.area[w] for w in newNeuron)
#         newLayer.append(newNeuron)
#     ax_coeff.append(newLayer)
# #print ("ax_coef",ax_coeff)
# print ("ax2:", area_AC, area_AX, area_AX/area_AC)

# mlpfx_ax=mlp_fxp_ps(ax_coeff,intercepts,inpfxp,wfxp,L0trunc,y_train,model)
# pred=mlpfx_ax.get_accuracy(X_test,y_test)
# predLst.append(pred)
# axLst.append(mlpfx_ax)
# print("ACCURACY W/ MLPFX AX2", pred)

# k=np.argmax(predLst)
# print("Using AX%d" % k)

# f=open("wax.v", 'w')
# actsizelst,outsize=write_mlp_verilog(f,inpfxp.get_width(), L0trunc, wfxp.get_width(), intercepts, axLst[k].coefs)
