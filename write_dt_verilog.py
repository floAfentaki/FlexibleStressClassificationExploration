import math
import random
from numpy import binary_repr
import sys

INPNAME="inp"
OUTNAME="out"
PRED="predo"

# comment out if predo=PRED is needed or regresor!!!!

def get_width (a):
    return int(a).bit_length()

#L0trunc=0, I leave it here as a reminder that we may use it to truncate the output of the 1st relu
def write_dt_verilog (f, inp_width, L0trunc, w_width, int_lst, w_lst):
    stdoutbckp=sys.stdout
    sys.stdout=f
    WIDTH_A=inp_width
    WIDTH_W=w_width
    WIDTH_ACT=[WIDTH_A]
    WEIGHTS=w_lst
    INTERCEPTS=int_lst

    REGRESSOR=bool(len(WEIGHTS[-1])==1)
    INP_NUM=len(WEIGHTS[0][0])
    OUT_NUM=len(WEIGHTS[-1])
    
    
    SUMWIDTH=[]
    for i,layer in enumerate(WEIGHTS):
        maxInp=(1<<WIDTH_ACT[i])-1
        maxsum=0
        minsum=0
        for j,neuron in enumerate(layer):
            pos=sum(w for w in neuron if w>0)*maxInp+INTERCEPTS[i][j]
            neg=sum(w for w in neuron if w<0)*maxInp+INTERCEPTS[i][j]
            maxsum=max(maxsum,pos)
            minsum=min(minsum,neg)
        size=max(1+get_width(maxsum), get_width(minsum))
        size=max(size, WIDTH_ACT[i]+w_width)
        #print("**",maxsum,minsum,size)
        SUMWIDTH.append(size)
        if i == 0:
            WIDTH_ACT.append(size-1-L0trunc)
        else:
             WIDTH_ACT.append(size-1)
    if REGRESSOR:
        WIDTH_O=SUMWIDTH[-1]
    else:
        WIDTH_O=get_width(len(WEIGHTS[-1]))
    print("//weights:",WEIGHTS) 
    print("//intercepts:",INTERCEPTS) 
    print("//act size:", WIDTH_ACT)
    print("//pred num:", OUT_NUM)

    simoutsize=0
    if REGRESSOR:
        print("module top ("+INPNAME+", "+OUTNAME+");")
        print("input ["+str(INP_NUM*WIDTH_A-1)+":"+str(0)+"] " + INPNAME +";")
        print("output ["+str(WIDTH_O-1)+":"+str(0)+"] " + OUTNAME +";")
    else:
        print("module top ("+INPNAME+", "+OUTNAME+");")
        # print("module top ("+INPNAME+", "+PRED+", "+OUTNAME+");")
        print("input ["+str(INP_NUM*WIDTH_A-1)+":"+str(0)+"] " + INPNAME +";")
        # print("output ["+str(OUT_NUM*SUMWIDTH[-1]-1)+":"+str(0)+"] " + PRED +";")
        print("output ["+str(WIDTH_O-1)+":"+str(0)+"] " + OUTNAME +";")
    print()
    
    act_next=[]
    for i in range(INP_NUM):
        a=INPNAME+"["+str((i+1)*WIDTH_A-1)+":"+str(i*WIDTH_A)+"]"
        act_next.append(a)
        
    for j in range(len(WEIGHTS)):
        act=list(act_next)
        act_next=[]
        actfunc='relu'
        #if j == len(WEIGHTS) -1:
        #    actfunc='identity'
        #else:
        #    actfunc='relu'
        for i in range(len(WEIGHTS[j])):
            print("// layer: %d - neuron: %d" % (j,i) )
            prefix = "n_"+str(j)+"_"+str(i)
            nweights=WEIGHTS[j][i]
            pwidth=WIDTH_ACT[j]+WIDTH_W
            rwidth=WIDTH_ACT[j+1]
            swidth=SUMWIDTH[j]
            bias=INTERCEPTS[j][i]
            # writeneuron (prefix,bias,nweights,act,WIDTH_W,pwidth,swidth,rwidth,actfunc)
            act_next.append(prefix)
            print()
    
    # if REGRESSOR:
    #     out = "    assign "+OUTNAME+" = {"
    # else:
    #     out = "    assign "+PRED+" = {"
    
    # for o in act_next:
    #     out = out+o+","
    # print(out[:-1]+"};")
    
    if not REGRESSOR:
        vw=SUMWIDTH[-1]
        iw=WIDTH_O
        prefix="argmax"
        print("// argmax: %d classes, need %d bits" % (len(WEIGHTS[-1]),iw) )
        # out=argmax(prefix,act_next,vw,iw)
        print("    assign "+OUTNAME+" = " + out + ";")
    print()
    print("endmodule")
    sys.stdout=stdoutbckp
    return WIDTH_ACT, WIDTH_O