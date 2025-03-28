# import math
# import random
from numpy import binary_repr
import sys

INPNAME="inp"
OUTNAME="out"
# PRED="predo"

def get_width (a):
    return int(a).bit_length()

def writeClassifier(prefix,b,nweights,act,WIDTH_W,pwidth,swidth):
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
        # if (isRegr):
        #      print("    wire signed [%d:0] %s;" % ((swidth-1),prefix))
        #      print("    assign %s = %s;" % (prefix,nsum))
        # else:
        print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
        print("    assign %s = %s;" % (sumname,nsum))
        print("    wire %s;" % (prefix))
        print("    assign %s = %s[%d];" % (prefix, sumname, (swidth-1)))

def decisionMatrix (prefix, act_next, numClasses):
    print("// decisionMatrix inp: " + ', '.join(act_next))
    swidth=get_width(numClasses)
    ind=0
    for m in range(numClasses):
        for n in range(numClasses):
                if m<n:
                    name_mn=prefix+"_cmp_"+str(m)+"_"+str(n)
                    name_nm=prefix+"_cmp_"+str(n)+"_"+str(m)
                    print("    wire %s, %s;" % (name_mn,name_nm))
                    print("    assign %s = ~%s;" % (name_mn, act_next[ind]) )
                    print("    assign %s = %s;" % (name_nm, act_next[ind]) )
                    print()
                    ind=ind+1
    dmlst=[]
    for m in range(numClasses):
        sum_m=prefix+"_sum_"+str(m)
        dmlst.append(sum_m)
        sumprefix=prefix+"_cmp_"+str(m)+"_"
        sum_row=' + '.join([sumprefix+str(n) for n in range(numClasses) if n != m])
        print("    wire [%d:0] %s;" % ((swidth-1),sum_m))
        print("    assign %s = %s;" % (sum_m, sum_row) ) 
    print()
    return dmlst

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



def write_svm_verilog (f, inp_width, w_width, int_lst, w_lst, numClasses):
    stdoutbckp=sys.stdout
    sys.stdout=f
    WIDTH_A=inp_width
    WIDTH_W=w_width
    WIDTH_ACT=[WIDTH_A]
    WEIGHTS=w_lst
    INTERCEPTS=int_lst

    # REGRESSOR=bool(numClasses==1)
    INP_NUM=len(WEIGHTS[0])
    OUT_NUM=numClasses
    
    
    SUMWIDTH=[]
    #for i,layer in enumerate(WEIGHTS):
    maxInp=(1<<WIDTH_ACT[0])-1
    maxsum=0
    minsum=0
    for j,neuron in enumerate(w_lst):
            pos=sum(w for w in neuron if w>0)*maxInp+INTERCEPTS[j]
            neg=sum(w for w in neuron if w<0)*maxInp+INTERCEPTS[j]
            maxsum=max(maxsum,pos)
            minsum=min(minsum,neg)
    size=max(1+get_width(maxsum), get_width(minsum))
    size=max(size, WIDTH_ACT[0]+w_width)
    #print("**",maxsum,minsum,size)
    SUMWIDTH.append(size)
    WIDTH_ACT.append(size)
    # if REGRESSOR:
    #     WIDTH_O=SUMWIDTH[-1]
    # else:
    WIDTH_O=get_width(numClasses)
    if numClasses==2:
        WIDTH_O=1

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
    print("input ["+str(INP_NUM*WIDTH_A-1)+":"+str(0)+"] " + INPNAME +";")
    # print("output ["+str(OUT_NUM*WIDTH_O-1)+":"+str(0)+"] " + PRED +";")
    print("output ["+str(WIDTH_O-1)+":"+str(0)+"] " + OUTNAME +";")
    print()
    
    act_next=[]
    for i in range(INP_NUM):
        a=INPNAME+"["+str((i+1)*WIDTH_A-1)+":"+str(i*WIDTH_A)+"]"
        act_next.append(a)
        
    act=list(act_next)
    act_next=[]
    actfunc='relu'
    j=0
    for i in range(len(WEIGHTS)):
            print("// classifier: %d" % (i) )
            prefix = "n_"+str(j)+"_"+str(i)
            nweights=WEIGHTS[i]
            pwidth=WIDTH_ACT[j]+WIDTH_W
            #rwidth=WIDTH_ACT[j+1]
            swidth=SUMWIDTH[j]
            bias=INTERCEPTS[i]
            writeClassifier (prefix,bias,nweights,act,WIDTH_W,pwidth,swidth)
            act_next.append(prefix)
            print()
    # if REGRESSOR:
    #     out = "    assign "+OUTNAME+" = {"
    #     for o in act_next:
    #         out = out+o+","
    #     print(out[:-1]+"};")
    
    # if not REGRESSOR:
    prefix="dm"
    if numClasses>2:
        dm=decisionMatrix(prefix, act_next, numClasses)
    # out = "    assign "+PRED+" = {"
    # for o in dm:
    #     out = out+o+","
    # print(out[:-1]+"};")

        vw=get_width(numClasses)
        iw=get_width(numClasses)
        prefix="argmax"
        print("// argmax: %d classes, need %d bits" % (numClasses,iw) )
        out=argmax(prefix,dm,vw,iw)
        print("    assign "+OUTNAME+" = " + out + ";")
    else:
        vw=1
        iw=1
        prefix="argmax"
        print("// argmax: %d classes, need %d bits" % (numClasses,iw) )
        # out=argmax(prefix,dm,vw,iw)
        print(f"    assign {OUTNAME} = ({act_next[0]}>0)?1'b0:1'b1;")
    print()
    print("endmodule")
    sys.stdout=stdoutbckp
    return WIDTH_ACT, WIDTH_O

def main():
    if len(sys.argv)==2:
        f=open(sys.argv[1],'w')
    else:
        f=sys.stdout
    outClasses=6
    wa=4
    ww=8
    lint = [-142, -16, -150, -134, -35, -16, -16, -8, 18, 16, 17, 29, 16, 16, 16]
    lw = [[1, 6, 1, 5, 3, 4, -10, 7, 4, -3, -5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 7, 2, -3, 5, 4, -6, 4, 6, -6, -2], [1, 6, 0, 1, 6, 6, -2, 6, 6, -8, -8], [0, 2, -1, 1, 2, 2, 0, 3, 2, -1, -3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 9, 3, 3, 2, -3, 4, -6, 3, -1, -8], [-3, 3, -2, 8, 7, 0, 1, 3, 4, -7, -8], [-2, 4, 1, 0, 1, -2, 4, 1, -1, -3, -5], [-2, 3, 0, -2, 3, -2, 6, 4, 0, -5, -5], [6, -1, -4, 6, 2, 0, 4, -3, 6, -2, -7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] 
    write_svm_verilog(f, wa, ww, lint, lw, outClasses)


if __name__ == "__main__":
    main()
