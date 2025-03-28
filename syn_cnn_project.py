import sys
# from sklearn.model_selection import train_test_split
from joblib import load
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import accuracy_score
# import torch

import numpy as np
import math
# import pandas as pd
from convfxp import ConvFxp
# from mlp_fxp_ps import mlp_fxp_ps
# from write_mlp_verilog import write_mlp_verilog

# import tensorflow
from tensorflow.keras import layers, models
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
 for _, l in enumerate(transp):
    newn=list()
    for _,n in enumerate(l):
        neww=list()
        for _,k in enumerate(n):
            # print(type(k))
            newww=list()
            if isinstance(k, list):                
                for _,w in enumerate(k):
                    if toFP:
                        newww.append(fxp.to_float(w))
                    else:
                        newww.append(fxp.to_fixed(w))
                neww.append(newww)
            else:
                if toFP:
                    neww.append(fxp.to_float(k))
                else:
                    neww.append(fxp.to_fixed(k))
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


def create_cnn(num_filters, kernel_size, activation, neurons, input_shape=(4, 1), num_classes=2, optimizer='adam'):
    model = models.Sequential([
        layers.Conv1D(num_filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape, use_bias=True),  # 4 kernels, kernel size 3
        layers.Flatten(),
        layers.Dense(neurons, activation='relu', use_bias=True),  # Output for num_classes classes
        layers.Dense(num_classes, activation='linear', use_bias=True)  # Output for num_classes classes
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def calculate_output_bitiwdth(inp_bitwidth, weights, num_sums):
        
    max_weight_bitwdth = max(int(x) for x in weights.flatten())
    max_product_value = (2**inp_bitwidth - 1) * max_weight_bitwdth
    max_sum_value = num_sums * max_product_value
    bitwdth = math.ceil(math.log2(max_sum_value))

    return bitwdth

def generate_verilog(conv_weights, dense_0_weights, dense_1_weights, ACT_FUNC, fxp_biases, NUM_INPUT=4, INP_BITWIDTH=4):

    kernel_size, _, num_kernels =np.shape(conv_weights)
    num_flat_kernels, num_neurons = np.shape(dense_0_weights)
    num_flat_kernels2, num_neurons2 = np.shape(dense_1_weights)

    print(kernel_size, num_kernels)
    print(num_flat_kernels, num_neurons)
    print(num_flat_kernels2, num_neurons2)

    CONV_BITWIDTH=calculate_output_bitiwdth(INP_BITWIDTH, conv_weights, kernel_size)
    DENSE_OUT_0_BITWIDTH=calculate_output_bitiwdth(CONV_BITWIDTH, dense_0_weights, num_flat_kernels)
    DENSE_OUT_1_BITWIDTH=calculate_output_bitiwdth(DENSE_OUT_0_BITWIDTH, dense_1_weights, num_flat_kernels2)
    WIDTH_OUT=math.ceil(math.log2(num_neurons))

    module_v=f"""
// {fxp_biases}
module top(
    input [{NUM_INPUT*INP_BITWIDTH-1}:0] inp,
    output [{WIDTH_OUT-1}:0] out  
    );"""

    
    conv_v=f"\n\n\t// Convolution Layer"
    conv_v+=f"\n\twire signed [{CONV_BITWIDTH-1}:0] conv_out[{num_flat_kernels-1}:0];"
    cnt=0
    for i in range(num_kernels):
        # conv_v+=f"\n\tassign conv_out[{i}] ="
        for l in range(num_kernels):
            conv_v+=f"\n\tassign conv_out[{cnt}] ="
            cnt+=1

            for j in range(kernel_size):
                conv_v+=f" {conv_weights[j, 0, l]}*inp[{(i + j) * INP_BITWIDTH + (INP_BITWIDTH-1)}:{(i + j) * INP_BITWIDTH}] +"
            conv_v=conv_v[0:-2] # Remove the last '+' sign
            conv_v+=";"
        conv_v+="\n"
        if cnt==num_flat_kernels:
            break

    act_v=f"\n\t// Activation Function: {ACT_FUNC}"
    if ACT_FUNC=='linear':
        act_v+=f"\n\twire signed [{CONV_BITWIDTH-1}:0] act_out[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels):
            act_v+=f"\n\tassign act_out[{i}] = conv_out[{i}];"
    elif ACT_FUNC=='relu':
        act_v+=f"\n\twire [{CONV_BITWIDTH-1-1}:0] act_out[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels):
            act_v+=f"\n\tassign act_out[{i}] = (conv_out[{i}][{CONV_BITWIDTH-1}]==1)? 0 : conv_out[{i}][{CONV_BITWIDTH-1-1}:0];"
    act_v+="\n"


    fc_0v="\n\t// Fully Connected "
    for i in range(num_neurons):
        fc_0v+=f"\n\twire signed [{DENSE_OUT_0_BITWIDTH-1}:0] dense_0_{i};"
        fc_0v+=f"\n\tassign dense_0_{i} ="
        for j in range(num_flat_kernels):
            fc_0v+=f" {dense_0_weights[j,i]}*act_out[{j}] +"
        fc_0v=fc_0v[0:-2] # Remove the last '+' sign
        fc_0v+=";\n" 


    act_0v=f"\n\t// Activation Function: {ACT_FUNC}"
    if ACT_FUNC=='linear':
        act_0v+=f"\n\twire signed [{DENSE_OUT_0_BITWIDTH-1}:0] act_out_2[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels2):
            act_0v+=f"\n\tassign act_out_2[{i}] = dense_0_{i};"
    elif ACT_FUNC=='relu':
        act_0v+=f"\n\twire [{DENSE_OUT_0_BITWIDTH-1-1}:0] act_out_2[{num_flat_kernels-1}:0];"
        for i in range(num_flat_kernels2):
            act_0v+=f"\n\tassign act_out_2[{i}] = (dense_0_{i}[{DENSE_OUT_0_BITWIDTH-1}]==1)? 0 : dense_0_{i}[{DENSE_OUT_0_BITWIDTH-1-1}:0];"
    act_0v+="\n"

    fc_1v="\n\t// Fully Connected "
    for i in range(num_neurons2):
        fc_1v+=f"\n\twire signed [{DENSE_OUT_1_BITWIDTH-1}:0] dense_1_{i};"
        fc_1v+=f"\n\tassign dense_1_{i} ="
        for j in range(num_flat_kernels2):
            fc_1v+=f" {dense_1_weights[j,i]}*act_out_2[{j}] +"
        fc_1v=fc_1v[0:-2] # Remove the last '+' sign
        fc_1v+=";\n" 

    # Argmax Logic
    argmax_v = "\n\n\t// Argmax Logic\n"
    for i in range(num_neurons2 - 1):
        argmax_v += f"\twire argmax_{i};\n"
        argmax_v += f"\tassign argmax_{i} = (dense_1_{i} > dense_1_{i+1}) ? {i} : {i+1};\n"

    argmax_v += "\tassign out = argmax_0"
    for i in range(1, num_neurons2 - 1):
        argmax_v += f" > dense_1_{i+1} ? argmax_{i} : {i+1}"

    argmax_v += ";"
    
    verilog_code = module_v+conv_v+act_v+fc_0v+act_0v+fc_1v+argmax_v+"\n\nendmodule"
    return verilog_code, WIDTH_OUT

dataset_name = sys.argv[1]
float_path = sys.argv[2]
number = sys.argv[3]
mfile = sys.argv[4]


X_train = np.load(float_path+"/X_train_30.npy")
X_test = np.load(float_path+"/X_test_30.npy")
y_train = np.load(float_path+"/y_train.npy")
y_test = np.load(float_path+"/y_test.npy")

X_train=X_train[:,:int(number.split('_')[0])]
X_test=X_test[:,:int(number.split('_')[0])]

# print(mfile)
loaded_model = load(mfile)
model = loaded_model.model
#dataset max size for sim
SIZE = 5000
#4 bit fraction for inputs
INP_FRAC=4
#8 bit weights
W_WIDTH=8
# trunc 1st relu
L0trunc=0

# trlst=[0,L0trunc]
# trlst=[0,L0trunc,L0trunc]

inpfxp=ConvFxp(0,0,INP_FRAC)

#I use the same representation for all the weights --> suboptimal
weights0=model.layers[0].get_weights()[0]
bias0=model.layers[0].get_weights()[1]
weights1=model.layers[2].get_weights()[0]
bias1=model.layers[2].get_weights()[1]
weights3=model.layers[3].get_weights()[0]
bias3=model.layers[3].get_weights()[1]
coefs=[weights0,weights1, weights3]
biases=[bias0,bias1, bias3]
w_int=get_width(get_maxabs(coefs))
w_frac=W_WIDTH-1-w_int
wfxp=ConvFxp(1,w_int,w_frac)

b_int=get_width(get_maxabs(biases))
b_frac=W_WIDTH-1-b_int
bfxp=ConvFxp(1,b_int,b_frac)

#I use no bias just fix the scripts!!
bfxp=[]
trlst=[0]
for i,c in enumerate(biases):
    trlst.append(L0trunc)
    b_int=get_width(get_maxabs(c))
    bfxp.append(ConvFxp(1,b_int,inpfxp.frac+(i+1)*wfxp.frac-trlst[i]))


print("Input fxp:", inpfxp.get_fxp())
print("Weights fxp:", wfxp.get_fxp())
print("Intercepts fxp:", [bfxp[i].get_fxp() for i in range(len(bfxp))])
print("Truncate HL:", L0trunc)

fxp_biases=convertIntercepts(biases,bfxp,False)
fxp_coeffs=convertCoef(coefs,wfxp,False)
fxp_conv_weights = np.transpose(fxp_coeffs[0], (2, 1, 0))
fxp_dense_0_weights = np.transpose(fxp_coeffs[1])
fxp_dense_1_weights = np.transpose(fxp_coeffs[2])
print(np.shape(fxp_coeffs[1]))
print(np.shape(coefs[1]))

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
# print(np.shape(pred))
# print(np.shape(y_test))
# print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))

y_test=[np.argmax(y, axis=None, out=None) for y in y_test[:]]
pred=[np.argmax(y, axis=None, out=None) for y in pred[:]]
y_train=[np.argmax(y, axis=None, out=None) for y in y_train[:]]
print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))

# mlpfx=mlp_fxp_ps(coefficients,intercepts,inpfxp,wfxp,L0trunc,y_train,model)
# mlpfx.get_accuracy(X_test,y_test)
# print("ACCURACY W/ MLPFX", mlpfx.get_accuracy(X_test,y_test))

f=open(f"{dataset_name}exact.v", 'w')
verilog_code, WIDTH_OUT=generate_verilog( fxp_conv_weights, fxp_dense_0_weights, fxp_dense_1_weights, 'relu', fxp_biases,  len(X_train[0]), INP_FRAC)
stdoutbckp=sys.stdout
sys.stdout=f
print(verilog_code)
sys.stdout=stdoutbckp
# quit()
# actsizelst,outsize=write_mlp_verilog(f,inpfxp.get_width(), L0trunc, wfxp.get_width(), intercepts, coefficients)
# print("Bitwidth of activations:", actsizelst[:-1] )

#write sim datasets
n=min(SIZE,len(X_train))
printData(X_train[0:n], inpfxp, f"{dataset_name}sim.Xtrain")
printData(X_test, inpfxp, f"{dataset_name}sim.Xtest")
f=open(f"{dataset_name}sim.Ytest","w")
np.savetxt(f,np.array(y_test).astype(int),fmt='%d',delimiter='\n')
# np.savetxt(f,Y_test.astype(int),fmt='%d',delimiter='\n')
f.close()
# dump(y_test, "sim.Ytest")

_, num_neurons = np.shape(fxp_dense_0_weights)

## testbench
f=open(f"{dataset_name}top_tb.v","w")
stdoutbckp=sys.stdout
sys.stdout=f
width_a=INP_FRAC
inp_num=len(X_test[0])
width_o=WIDTH_OUT
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
conf.append('export PRED_NUM='+str(len(fxp_dense_1_weights[-1])))
conf.append('#unique values of Ytrain')
conf.append('SIM_YLST='+uniqYtrain)
conf.append('SIM_REGRESSOR="'+isRegr+'";# True or False')
conf.append('#number of fractional bits in the output')
# conf.append('SIM_FRACTION_BITS='+str(mlpfx.Q))
conf.append('SIM_WIDTH_A="'+str(inpfxp.get_width())+'"')
conf.append('SIM_NUM_A="'+str(len(fxp_conv_weights[0][0]))+'"')
conf.append('SIM_OUTWIDTH="'+str(width_o)+'"')
f.write('\n'.join(conf) + '\n')
f.close()
