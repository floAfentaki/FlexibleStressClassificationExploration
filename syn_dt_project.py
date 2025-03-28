import sys
import numpy as np
import math
from convfxp import ConvFxp
from joblib import load
from sklearn import tree
from sklearn.metrics import accuracy_score


synth_period=200000.00
# COMP_WIDTH=8

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
COMP_WIDTH = int(sys.argv[5])


X_train = np.load(float_path+"/X_train_30.npy")
X_test = np.load(float_path+"/X_test_30.npy")
y_train = np.load(float_path+"/y_train.npy")
y_test = np.load(float_path+"/y_test.npy")

X_train=X_train[:,:int(number.split('_')[0])]
X_test=X_test[:,:int(number.split('_')[0])]

model = load(mfile)

pred=model.predict(X_test)
isRegr="False"
predname="predo"
if "MLPRegressor" in str(type(model)):
        predname="out"
        isRegr="True"
        pred=np.clip(np.round(pred),min(y_test),max(y_test))
print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))

# Function to extract rules
def get_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    print("assign out=",end='')

    def recurse(left, right, threshold, features, node, depth=0):
        indent = "  " * depth
        threshold=(np.round(threshold*2**COMP_WIDTH))/2**COMP_WIDTH
        # print(threshold)
        if (threshold[node] != -2):
            # print(f"{indent}if ( {features[node]}<<8 <= 8'd{int(threshold[node]*(2**8))} ) begin")
            # print(f"{indent} ( {features[node]}<<8 <= 8'd{int(threshold[node]*(2**8))} ) ?")
            print(f"( {features[node]} <= {COMP_WIDTH}'d{int(threshold[node]*(2**COMP_WIDTH))} ) ?", end='')
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth+1)
            # print(f"{indent}else: begin")
            print(":",end='')
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], depth+1)
        else:
            # print(f"{indent}out={np.argmax(value[node])}; end")
            # print(f"{indent}{np.argmax(value[node])}")
            print(f"{np.argmax(value[node])}", end='')
    
    recurse(left, right, threshold, features, 0)
    print(";")

# INP_NUM=len([i for i in model.tree_.feature  if (i != -2)])
INP_NUM=int(number.split('_')[0])
OUT_NUM=len(model.tree_.n_classes)


#dataset max size for sim
SIZE = 5000
inpfxp=ConvFxp(0,0,COMP_WIDTH)

pred=model.predict(X_test)
isRegr="False"
predname="predo"
if "MLPRegressor" in str(type(model)):
        predname="out"
        isRegr="True"
        pred=np.clip(np.round(pred),min(y_test),max(y_test))
print("ACCURACY W/ MODEL:", accuracy_score(pred, y_test))


f=open(f"{dataset_name}exact.v", 'w')
stdoutbckp=sys.stdout
sys.stdout=f
print("module top (inp, out);")
print(f"input [{(INP_NUM*COMP_WIDTH)-1}:0] inp;")
print(f"output [{get_width(OUT_NUM)}:0] out;")
get_rules(model, [f"inp[{i*COMP_WIDTH+COMP_WIDTH-1}:{i*COMP_WIDTH}]" for i in range(len(X_train))])
print("endmodule")
sys.stdout=stdoutbckp

#write sim datasets
n=min(SIZE,len(X_train))
printData(X_train[0:n], inpfxp, f"{dataset_name}sim.Xtrain")
printData(X_test, inpfxp, f"{dataset_name}sim.Xtest")
f=open(f"{dataset_name}sim.Ytest","w")
np.savetxt(f,y_test.astype(int),fmt='%d',delimiter='\n')
f.close()

## testbench
f=open(f"{dataset_name}top_tb.v","w")
stdoutbckp=sys.stdout
sys.stdout=f
width_a=COMP_WIDTH
inp_num=len(X_test[0])
width_o=get_width(OUT_NUM)
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
conf.append('#unique values of Ytrain')
conf.append('SIM_YLST='+uniqYtrain)
conf.append('SIM_REGRESSOR="'+isRegr+'";# True or False')
conf.append('#number of fractional bits in the output')
conf.append('SIM_WIDTH_A="'+str(inpfxp.get_width())+'"')
f.write('\n'.join(conf) + '\n')
f.close()