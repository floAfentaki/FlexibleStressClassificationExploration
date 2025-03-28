import math
import csv
import sys
from numpy import binary_repr

INPNAME = "inp"
OUTNAME = "out"

def get_width(a):
    return int(a).bit_length()

def writeneuron_module(neuron_id, nweights, bias, actfunc, WIDTH_ACT, WIDTH_W, swidth, rwidth):
    # Create neuron module for each neuron
    print(f"module neuron_{neuron_id} (input signed [{WIDTH_ACT-1}:0] " +
          ', '.join([f"in_{i}" for i in range(len(nweights))]) + ",")
    print(f"                  output signed [{rwidth-1}:0] out);")
    
    # Define the weighted sum
    nsum = "sum"
    print(f"    wire signed [{swidth-1}:0] {nsum};")
    
    # Loop over the inputs and weights and calculate product
    terms = []
    for i, w in enumerate(nweights):
        bin_w = f"{WIDTH_W}'sb" + binary_repr(w, WIDTH_W)
        prod_name = f"prod_{i}"
        print(f"    wire signed [{swidth-1}:0] {prod_name};")
        print(f"    assign {prod_name} = $signed({{1'b0, in_{i}}}) * {bin_w};")
        terms.append(prod_name)
    
    # Sum of products with bias
    print(f"    assign {nsum} = {bias} + " + ' + '.join(terms) + ";")
    
    # Apply activation function (ReLU or identity)
    if actfunc == 'relu':
        print(f"    assign out = ({nsum} < 0) ? {rwidth}'d0 : {nsum}[{rwidth-1}:0];")
    else:
        print(f"    assign out = {nsum}[{rwidth-1}:0];")
    
    print("endmodule\n")

def argmax(prefix, act, vwidth, iwidth):
    lvl = 0
    vallist = list(act)
    print(f"// argmax inp: {', '.join(vallist)}")
    idxlist = [f"{iwidth}'b" + binary_repr(i, iwidth) for i in range(len(act))]
    
    while len(vallist) > 1:
        newV = []
        newI = []
        print(f"    //comp level {lvl}")
        for i in range(0, len(vallist) - 1, 2):
            cmpname = f"cmp_{lvl}_{i}"
            vname = f"{prefix}_val_{lvl}_{i}"
            iname = f"{prefix}_idx_{lvl}_{i}"
            vname1 = vallist[i]
            vname2 = vallist[i + 1]
            iname1 = idxlist[i]
            iname2 = idxlist[i + 1]
            
            print(f"    wire {cmpname};")
            print(f"    wire [{vwidth-1}:0] {vname};")
            print(f"    wire [{iwidth-1}:0] {iname};")
    
            print(f"    assign {cmpname} = ( {vname1} >= {vname2} );")
            print(f"    assign {vname} = ( {cmpname} ) ? {vname1} : {vname2};")
            print(f"    assign {iname} = ( {cmpname} ) ? {iname1} : {iname2};\n")
            
            newV.append(vname)
            newI.append(iname)
        
        if len(vallist) % 2 == 1:
            newV.append(vallist[-1])
            newI.append(idxlist[-1])
        
        lvl += 1
        vallist = newV
        idxlist = newI
    
    return idxlist[-1]

def write_mlp_verilog(f, inp_width, L0trunc, w_width, int_lst, w_lst):
    stdoutbckp = sys.stdout
    sys.stdout = f
    print(f"//{w_lst}")
    print(f"//{int_lst}")
    WIDTH_A = inp_width
    WIDTH_W = w_width
    WIDTH_ACT = [WIDTH_A]
    WEIGHTS = w_lst
    INTERCEPTS = int_lst

    INP_NUM = len(WEIGHTS[0][0])
    OUT_NUM = len(WEIGHTS[-1])
    
    SUMWIDTH = []
    for i, layer in enumerate(WEIGHTS):
        maxInp = (1 << WIDTH_ACT[i]) - 1
        maxsum = 0
        minsum = 0
        for j, neuron in enumerate(layer):
            pos = sum(w for w in neuron if w > 0) * maxInp + INTERCEPTS[i][j]
            neg = sum(w for w in neuron if w < 0) * maxInp + INTERCEPTS[i][j]
            maxsum = max(maxsum, pos)
            minsum = min(minsum, neg)
        size = max(1 + get_width(maxsum), get_width(minsum))
        size = max(size, WIDTH_ACT[i] + w_width)
        SUMWIDTH.append(size)
        WIDTH_ACT.append(size - 1)
    
    WIDTH_O = get_width(len(WEIGHTS[-1]))
    
    # Neuron modules
    modules=[]
    for j, layer in enumerate(WEIGHTS):
        for i, nweights in enumerate(layer):
            bias = INTERCEPTS[j][i]
            swidth = SUMWIDTH[j]
            rwidth = WIDTH_ACT[j + 1]
            modules.append(f"neuron_n_{j}_{i}")
            writeneuron_module(f"n_{j}_{i}", nweights, bias, 'relu', WIDTH_ACT[j], WIDTH_W ,swidth, rwidth)
    
    # Top module
    print(f"module top (input [{INP_NUM * WIDTH_A - 1}:0] {INPNAME}, output [{WIDTH_O-1}:0] {OUTNAME});")
    
    # Split the input vector into individual inputs
    for i in range(INP_NUM):
        print(f"    wire signed [{WIDTH_A-1}:0] in_{i} = {INPNAME}[{(i+1) * WIDTH_A - 1}:{i * WIDTH_A}];")
    
    act_next = []
    for j, layer in enumerate(WEIGHTS):
        act = list(act_next)
        act_next = []
        for i in range(len(layer)):
            prefix = f"n_{j}_{i}"
            inputs = ', '.join([f"in_{k}" for k in range(len(WEIGHTS[j][i]))])
            print(f"    wire signed [{WIDTH_ACT[j+1]-1}:0] {prefix}_out;")
            if j == 0:
                print(f"    neuron_{prefix} neuron_{prefix}_inst ({inputs}, {prefix}_out);")
            else:
                prev_outputs = ', '.join([f"{act[k]}" for k in range(len(act))])
                print(f"    neuron_{prefix} neuron_{prefix}_inst ({prev_outputs}, {prefix}_out);")
            act_next.append(f"{prefix}_out")
    
    vw = SUMWIDTH[-1]
    iw = WIDTH_O
    prefix = "argmax"
    print(f"// argmax: {len(WEIGHTS[-1])} classes, need {iw} bits")
    out = argmax(prefix, act_next, vw, iw)
    print(f"    assign {OUTNAME} = {out};")
    print(f"endmodule")
    
    sys.stdout = stdoutbckp
    print(modules)
    with open('modules.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(modules)

    return WIDTH_ACT, WIDTH_O
