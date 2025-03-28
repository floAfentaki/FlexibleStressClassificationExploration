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

def writeneuron(prefix,b,nweights,act,WIDTH_W,pwidth,swidth,rwidth,actfunc):
        nsum=""+str(b)
        sumname = prefix+"_sum"
        for i in range(len(nweights)):
            w=nweights[i]
            a=act[i]
            name=prefix+"_po_"+str(i)
            nsum = nsum+" + "+ name
            print("    wire signed [%d:0] %s;" % ((pwidth-1),name))
            bin_w=str(WIDTH_W)+"'sb"+binary_repr(w,WIDTH_W)
            print("    //weight %d: %s" % (w, bin_w))
            print("    assign %s = $signed({1'b0, %s}) * %s;" % (name,a,bin_w))
            print()
        print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
        print("    assign %s = %s;" % (sumname,nsum))
        print("    //%s" % actfunc)
        if actfunc == 'relu':
            print("    wire [%d:0] %s;" % ((rwidth-1),prefix))
            print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s[%d:%d]);" % (prefix,sumname,rwidth,sumname,(swidth-1-1),(swidth-1-rwidth)))
        else:
            print("    wire signed [%d:0] %s;" % ((rwidth-1),prefix))
            print("    assign %s = %s[%d:0];" % (prefix,sumname,(rwidth-1)))

def argmax (prefix,act,vwidth,iwidth):
        lvl=0
        vallist=list(act)
        print("// argmax inp: " + ', '.join(vallist))
        idxlist=[str(iwidth)+"'b"+binary_repr(i,iwidth) for i in range(len(act))]
        while len(vallist) > 1:
            newV=[]
            newI=[]
            comp=0
            print("    //comp level %d" % lvl)
            for i in range(0,len(vallist)-1,2):
                cmpname="cmp_"+str(lvl)+"_"+str(i)
                vname=prefix+"_val_"+str(lvl)+"_"+str(i)
                iname=prefix+"_idx_"+str(lvl)+"_"+str(i)
                vname1=vallist[i]
                vname2=vallist[i+1]
                iname1=idxlist[i]
                iname2=idxlist[i+1]
                print("    wire %s;" % cmpname)
                print("    wire [%d:0] %s;" % ((vwidth-1),vname))
                print("    wire [%d:0] %s;" % ((iwidth-1),iname))
    
                print("    assign {%s} = ( %s >= %s );" % (cmpname,vname1,vname2))
                print("    assign {%s} = ( %s ) ? %s : %s;" % (vname,cmpname,vname1,vname2))
                print("    assign {%s} = ( %s ) ? %s : %s;" % (iname,cmpname,iname1,iname2))
                print()
                newV.append(vname)
                newI.append(iname)
            if len(vallist) % 2 == 1:
                newV.append(vallist[-1])
                newI.append(idxlist[-1])
            lvl+=1
            vallist = list(newV)
            idxlist = list(newI)
        return idxlist[-1]



#L0trunc=0, I leave it here as a reminder that we may use it to truncate the output of the 1st relu
def write_mlp_verilog (f, inp_width, L0trunc, w_width, int_lst, w_lst):
    stdoutbckp=sys.stdout
    sys.stdout=f
    WIDTH_A=inp_width
    WIDTH_W=w_width
    WIDTH_ACT=[WIDTH_A]
    WEIGHTS=w_lst
    INTERCEPTS=int_lst

    # REGRESSOR=bool(len(WEIGHTS[-1])==1)
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
    # if REGRESSOR:
    #     WIDTH_O=SUMWIDTH[-1]
    # else:
    WIDTH_O=get_width(len(WEIGHTS[-1]))
    print("//weights:",WEIGHTS) 
    print("//intercepts:",INTERCEPTS) 
    print("//act size:", WIDTH_ACT)
    print("//pred num:", OUT_NUM)

    simoutsize=0
    # if REGRESSOR:
    #     print("module top ("+INPNAME+", "+OUTNAME+");")
    #     print("input ["+str(INP_NUM*WIDTH_A-1)+":"+str(0)+"] " + INPNAME +";")
    #     print("output ["+str(WIDTH_O-1)+":"+str(0)+"] " + OUTNAME +";")
    # else:
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
            writeneuron (prefix,bias,nweights,act,WIDTH_W,pwidth,swidth,rwidth,actfunc)
            act_next.append(prefix)
            print()
    
    # if REGRESSOR:
    #     out = "    assign "+OUTNAME+" = {"
    # else:
    #     out = "    assign "+PRED+" = {"
    
    # for o in act_next:
    #     out = out+o+","
    # print(out[:-1]+"};")
    
    # if not REGRESSOR:
    vw=SUMWIDTH[-1]
    iw=WIDTH_O
    prefix="argmax"
    print("// argmax: %d classes, need %d bits" % (len(WEIGHTS[-1]),iw) )
    out=argmax(prefix,act_next,vw,iw)
    print("    assign "+OUTNAME+" = " + out + ";")
    print()
    print("endmodule")
    sys.stdout=stdoutbckp
    return WIDTH_ACT, WIDTH_O
'''
def main():
    if len(sys.argv)==2:
        f=open(sys.argv[1],'w')
    else:
        f=sys.stdout
    wa=4
    ww=8
    lint = [[-20, 1426, -138, -292, 206, 754], [5777, 6097, 7582, 7455, -6210, -20997]]
    lw = [[[-10, -4, -1, 0, -2, -6, -3, -5, -5, 6, -10], [-12, -40, 7, -60, -75, -36, 12, 26, -31, 60, -28], [-1, -8, -3, 6, 1, 1, -4, 3, -3, -1, -2], [-8, -8, -10, 0, 4, -6, -5, -7, 1, 2, -10], [-30, -27, -63, -31, 24, 36, 93, 56, -41, -4, -101], [-27, 17, 20, -49, -30, -27, 23, 33, -15, -3, -64]], [[-4, -46, -1, -2, -53, 53], [8, -18, 9, -10, 35, 20], [-6, -9, 0, -10, 58, 22], [-4, 5, -12, -2, 44, -5], [-2, 26, 0, 1, -38, -38], [11, 33, -11, -6, -13, -48]]]
    write_mlp_verilog(f, wa, 4, ww, lint, lw)


if __name__ == "__main__":
    main()
'''
