//weights: [[[24, -19, -7, -15, -4], [4, 18, -5, 24, -13], [0, 16, -15, -5, -7], [-20, -25, 1, 11, -12], [16, -9, -1, -9, -20], [-22, -18, -8, -10, 18], [20, 6, -24, 9, -24], [20, -17, 8, -3, -2], [19, 3, 20, -1, -11], [1, 0, -1, -1, -1], [-21, -21, 19, 29, 2], [11, 18, 6, -12, 3], [-1, -18, 28, 23, 24], [2, 18, 2, 14, -2], [0, 0, 0, 0, 0], [28, 22, 7, -28, 0], [5, 12, -14, -4, 5], [-7, 4, -1, -4, -4], [0, 0, -3, -3, -3], [-1, 0, -2, -2, -2]], [[5, 16, 10, 0, -8, 0, 10, 30, -15, 0, -16, -14, -16, 20, 0, 31, -19, -24, 0, 0], [23, 16, -88, 15, 0, 15, -25, -11, -8, -1, -13, 33, -8, -20, 0, 8, 0, 0, 2, 0], [11, 23, -22, -18, 27, -17, 20, -22, 24, 0, -5, -7, -18, -6, 0, 1, 22, 20, 0, 0]]]
//intercepts: [[4740, 7450, -4711, 4252, 1940, -255, 2811, 1083, 1672, 6342, -4595, -273, -876, 4728, -1203, 8192, -350, 3840, 2830, 478], [-256537, 43699, 213949]]
//act size: [8, 15, 23]
//pred num: 3
//sum size: [16, 24]
module top (input [39:0] inp, output [1:0] out);
    // Layer 0, Neuron 0
    wire signed [15:0] n_0_0_po_0;
    // weight 24: 8'sb00011000
    assign n_0_0_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00011000;

    wire signed [15:0] n_0_0_po_1;
    // weight -19: 8'sb11101101
    assign n_0_0_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11101101;

    wire signed [15:0] n_0_0_po_2;
    // weight -7: 8'sb11111001
    assign n_0_0_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111001;

    wire signed [15:0] n_0_0_po_3;
    // weight -15: 8'sb11110001
    assign n_0_0_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11110001;

    wire signed [15:0] n_0_0_po_4;
    // weight -4: 8'sb11111100
    assign n_0_0_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111100;

    wire signed [15:0] n_0_0_sum;
    assign n_0_0_sum = 4740 + n_0_0_po_0 + n_0_0_po_1 + n_0_0_po_2 + n_0_0_po_3 + n_0_0_po_4;
    // relu
    wire [14:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_0_sum[14:0]);

    // Layer 0, Neuron 1
    wire signed [15:0] n_0_1_po_0;
    // weight 4: 8'sb00000100
    assign n_0_1_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000100;

    wire signed [15:0] n_0_1_po_1;
    // weight 18: 8'sb00010010
    assign n_0_1_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00010010;

    wire signed [15:0] n_0_1_po_2;
    // weight -5: 8'sb11111011
    assign n_0_1_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111011;

    wire signed [15:0] n_0_1_po_3;
    // weight 24: 8'sb00011000
    assign n_0_1_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00011000;

    wire signed [15:0] n_0_1_po_4;
    // weight -13: 8'sb11110011
    assign n_0_1_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11110011;

    wire signed [15:0] n_0_1_sum;
    assign n_0_1_sum = 7450 + n_0_1_po_0 + n_0_1_po_1 + n_0_1_po_2 + n_0_1_po_3 + n_0_1_po_4;
    // relu
    wire [14:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_1_sum[14:0]);

    // Layer 0, Neuron 2
    wire signed [15:0] n_0_2_po_0;
    // weight 0: 8'sb00000000
    assign n_0_2_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000000;

    wire signed [15:0] n_0_2_po_1;
    // weight 16: 8'sb00010000
    assign n_0_2_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00010000;

    wire signed [15:0] n_0_2_po_2;
    // weight -15: 8'sb11110001
    assign n_0_2_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11110001;

    wire signed [15:0] n_0_2_po_3;
    // weight -5: 8'sb11111011
    assign n_0_2_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111011;

    wire signed [15:0] n_0_2_po_4;
    // weight -7: 8'sb11111001
    assign n_0_2_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111001;

    wire signed [15:0] n_0_2_sum;
    assign n_0_2_sum = -4711 + n_0_2_po_0 + n_0_2_po_1 + n_0_2_po_2 + n_0_2_po_3 + n_0_2_po_4;
    // relu
    wire [14:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_2_sum[14:0]);

    // Layer 0, Neuron 3
    wire signed [15:0] n_0_3_po_0;
    // weight -20: 8'sb11101100
    assign n_0_3_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11101100;

    wire signed [15:0] n_0_3_po_1;
    // weight -25: 8'sb11100111
    assign n_0_3_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11100111;

    wire signed [15:0] n_0_3_po_2;
    // weight 1: 8'sb00000001
    assign n_0_3_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00000001;

    wire signed [15:0] n_0_3_po_3;
    // weight 11: 8'sb00001011
    assign n_0_3_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00001011;

    wire signed [15:0] n_0_3_po_4;
    // weight -12: 8'sb11110100
    assign n_0_3_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11110100;

    wire signed [15:0] n_0_3_sum;
    assign n_0_3_sum = 4252 + n_0_3_po_0 + n_0_3_po_1 + n_0_3_po_2 + n_0_3_po_3 + n_0_3_po_4;
    // relu
    wire [14:0] n_0_3;
    assign n_0_3 = (n_0_3_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_3_sum[14:0]);

    // Layer 0, Neuron 4
    wire signed [15:0] n_0_4_po_0;
    // weight 16: 8'sb00010000
    assign n_0_4_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00010000;

    wire signed [15:0] n_0_4_po_1;
    // weight -9: 8'sb11110111
    assign n_0_4_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11110111;

    wire signed [15:0] n_0_4_po_2;
    // weight -1: 8'sb11111111
    assign n_0_4_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111111;

    wire signed [15:0] n_0_4_po_3;
    // weight -9: 8'sb11110111
    assign n_0_4_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11110111;

    wire signed [15:0] n_0_4_po_4;
    // weight -20: 8'sb11101100
    assign n_0_4_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11101100;

    wire signed [15:0] n_0_4_sum;
    assign n_0_4_sum = 1940 + n_0_4_po_0 + n_0_4_po_1 + n_0_4_po_2 + n_0_4_po_3 + n_0_4_po_4;
    // relu
    wire [14:0] n_0_4;
    assign n_0_4 = (n_0_4_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_4_sum[14:0]);

    // Layer 0, Neuron 5
    wire signed [15:0] n_0_5_po_0;
    // weight -22: 8'sb11101010
    assign n_0_5_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11101010;

    wire signed [15:0] n_0_5_po_1;
    // weight -18: 8'sb11101110
    assign n_0_5_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11101110;

    wire signed [15:0] n_0_5_po_2;
    // weight -8: 8'sb11111000
    assign n_0_5_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111000;

    wire signed [15:0] n_0_5_po_3;
    // weight -10: 8'sb11110110
    assign n_0_5_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11110110;

    wire signed [15:0] n_0_5_po_4;
    // weight 18: 8'sb00010010
    assign n_0_5_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00010010;

    wire signed [15:0] n_0_5_sum;
    assign n_0_5_sum = -255 + n_0_5_po_0 + n_0_5_po_1 + n_0_5_po_2 + n_0_5_po_3 + n_0_5_po_4;
    // relu
    wire [14:0] n_0_5;
    assign n_0_5 = (n_0_5_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_5_sum[14:0]);

    // Layer 0, Neuron 6
    wire signed [15:0] n_0_6_po_0;
    // weight 20: 8'sb00010100
    assign n_0_6_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00010100;

    wire signed [15:0] n_0_6_po_1;
    // weight 6: 8'sb00000110
    assign n_0_6_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000110;

    wire signed [15:0] n_0_6_po_2;
    // weight -24: 8'sb11101000
    assign n_0_6_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11101000;

    wire signed [15:0] n_0_6_po_3;
    // weight 9: 8'sb00001001
    assign n_0_6_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00001001;

    wire signed [15:0] n_0_6_po_4;
    // weight -24: 8'sb11101000
    assign n_0_6_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11101000;

    wire signed [15:0] n_0_6_sum;
    assign n_0_6_sum = 2811 + n_0_6_po_0 + n_0_6_po_1 + n_0_6_po_2 + n_0_6_po_3 + n_0_6_po_4;
    // relu
    wire [14:0] n_0_6;
    assign n_0_6 = (n_0_6_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_6_sum[14:0]);

    // Layer 0, Neuron 7
    wire signed [15:0] n_0_7_po_0;
    // weight 20: 8'sb00010100
    assign n_0_7_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00010100;

    wire signed [15:0] n_0_7_po_1;
    // weight -17: 8'sb11101111
    assign n_0_7_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11101111;

    wire signed [15:0] n_0_7_po_2;
    // weight 8: 8'sb00001000
    assign n_0_7_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00001000;

    wire signed [15:0] n_0_7_po_3;
    // weight -3: 8'sb11111101
    assign n_0_7_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111101;

    wire signed [15:0] n_0_7_po_4;
    // weight -2: 8'sb11111110
    assign n_0_7_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111110;

    wire signed [15:0] n_0_7_sum;
    assign n_0_7_sum = 1083 + n_0_7_po_0 + n_0_7_po_1 + n_0_7_po_2 + n_0_7_po_3 + n_0_7_po_4;
    // relu
    wire [14:0] n_0_7;
    assign n_0_7 = (n_0_7_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_7_sum[14:0]);

    // Layer 0, Neuron 8
    wire signed [15:0] n_0_8_po_0;
    // weight 19: 8'sb00010011
    assign n_0_8_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00010011;

    wire signed [15:0] n_0_8_po_1;
    // weight 3: 8'sb00000011
    assign n_0_8_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000011;

    wire signed [15:0] n_0_8_po_2;
    // weight 20: 8'sb00010100
    assign n_0_8_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00010100;

    wire signed [15:0] n_0_8_po_3;
    // weight -1: 8'sb11111111
    assign n_0_8_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111111;

    wire signed [15:0] n_0_8_po_4;
    // weight -11: 8'sb11110101
    assign n_0_8_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11110101;

    wire signed [15:0] n_0_8_sum;
    assign n_0_8_sum = 1672 + n_0_8_po_0 + n_0_8_po_1 + n_0_8_po_2 + n_0_8_po_3 + n_0_8_po_4;
    // relu
    wire [14:0] n_0_8;
    assign n_0_8 = (n_0_8_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_8_sum[14:0]);

    // Layer 0, Neuron 9
    wire signed [15:0] n_0_9_po_0;
    // weight 1: 8'sb00000001
    assign n_0_9_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000001;

    wire signed [15:0] n_0_9_po_1;
    // weight 0: 8'sb00000000
    assign n_0_9_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000000;

    wire signed [15:0] n_0_9_po_2;
    // weight -1: 8'sb11111111
    assign n_0_9_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111111;

    wire signed [15:0] n_0_9_po_3;
    // weight -1: 8'sb11111111
    assign n_0_9_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111111;

    wire signed [15:0] n_0_9_po_4;
    // weight -1: 8'sb11111111
    assign n_0_9_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111111;

    wire signed [15:0] n_0_9_sum;
    assign n_0_9_sum = 6342 + n_0_9_po_0 + n_0_9_po_1 + n_0_9_po_2 + n_0_9_po_3 + n_0_9_po_4;
    // relu
    wire [14:0] n_0_9;
    assign n_0_9 = (n_0_9_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_9_sum[14:0]);

    // Layer 0, Neuron 10
    wire signed [15:0] n_0_10_po_0;
    // weight -21: 8'sb11101011
    assign n_0_10_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11101011;

    wire signed [15:0] n_0_10_po_1;
    // weight -21: 8'sb11101011
    assign n_0_10_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11101011;

    wire signed [15:0] n_0_10_po_2;
    // weight 19: 8'sb00010011
    assign n_0_10_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00010011;

    wire signed [15:0] n_0_10_po_3;
    // weight 29: 8'sb00011101
    assign n_0_10_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00011101;

    wire signed [15:0] n_0_10_po_4;
    // weight 2: 8'sb00000010
    assign n_0_10_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00000010;

    wire signed [15:0] n_0_10_sum;
    assign n_0_10_sum = -4595 + n_0_10_po_0 + n_0_10_po_1 + n_0_10_po_2 + n_0_10_po_3 + n_0_10_po_4;
    // relu
    wire [14:0] n_0_10;
    assign n_0_10 = (n_0_10_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_10_sum[14:0]);

    // Layer 0, Neuron 11
    wire signed [15:0] n_0_11_po_0;
    // weight 11: 8'sb00001011
    assign n_0_11_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00001011;

    wire signed [15:0] n_0_11_po_1;
    // weight 18: 8'sb00010010
    assign n_0_11_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00010010;

    wire signed [15:0] n_0_11_po_2;
    // weight 6: 8'sb00000110
    assign n_0_11_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00000110;

    wire signed [15:0] n_0_11_po_3;
    // weight -12: 8'sb11110100
    assign n_0_11_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11110100;

    wire signed [15:0] n_0_11_po_4;
    // weight 3: 8'sb00000011
    assign n_0_11_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00000011;

    wire signed [15:0] n_0_11_sum;
    assign n_0_11_sum = -273 + n_0_11_po_0 + n_0_11_po_1 + n_0_11_po_2 + n_0_11_po_3 + n_0_11_po_4;
    // relu
    wire [14:0] n_0_11;
    assign n_0_11 = (n_0_11_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_11_sum[14:0]);

    // Layer 0, Neuron 12
    wire signed [15:0] n_0_12_po_0;
    // weight -1: 8'sb11111111
    assign n_0_12_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11111111;

    wire signed [15:0] n_0_12_po_1;
    // weight -18: 8'sb11101110
    assign n_0_12_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb11101110;

    wire signed [15:0] n_0_12_po_2;
    // weight 28: 8'sb00011100
    assign n_0_12_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00011100;

    wire signed [15:0] n_0_12_po_3;
    // weight 23: 8'sb00010111
    assign n_0_12_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00010111;

    wire signed [15:0] n_0_12_po_4;
    // weight 24: 8'sb00011000
    assign n_0_12_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00011000;

    wire signed [15:0] n_0_12_sum;
    assign n_0_12_sum = -876 + n_0_12_po_0 + n_0_12_po_1 + n_0_12_po_2 + n_0_12_po_3 + n_0_12_po_4;
    // relu
    wire [14:0] n_0_12;
    assign n_0_12 = (n_0_12_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_12_sum[14:0]);

    // Layer 0, Neuron 13
    wire signed [15:0] n_0_13_po_0;
    // weight 2: 8'sb00000010
    assign n_0_13_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000010;

    wire signed [15:0] n_0_13_po_1;
    // weight 18: 8'sb00010010
    assign n_0_13_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00010010;

    wire signed [15:0] n_0_13_po_2;
    // weight 2: 8'sb00000010
    assign n_0_13_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00000010;

    wire signed [15:0] n_0_13_po_3;
    // weight 14: 8'sb00001110
    assign n_0_13_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00001110;

    wire signed [15:0] n_0_13_po_4;
    // weight -2: 8'sb11111110
    assign n_0_13_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111110;

    wire signed [15:0] n_0_13_sum;
    assign n_0_13_sum = 4728 + n_0_13_po_0 + n_0_13_po_1 + n_0_13_po_2 + n_0_13_po_3 + n_0_13_po_4;
    // relu
    wire [14:0] n_0_13;
    assign n_0_13 = (n_0_13_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_13_sum[14:0]);

    // Layer 0, Neuron 14
    wire signed [15:0] n_0_14_po_0;
    // weight 0: 8'sb00000000
    assign n_0_14_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000000;

    wire signed [15:0] n_0_14_po_1;
    // weight 0: 8'sb00000000
    assign n_0_14_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000000;

    wire signed [15:0] n_0_14_po_2;
    // weight 0: 8'sb00000000
    assign n_0_14_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00000000;

    wire signed [15:0] n_0_14_po_3;
    // weight 0: 8'sb00000000
    assign n_0_14_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb00000000;

    wire signed [15:0] n_0_14_po_4;
    // weight 0: 8'sb00000000
    assign n_0_14_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00000000;

    wire signed [15:0] n_0_14_sum;
    assign n_0_14_sum = -1203 + n_0_14_po_0 + n_0_14_po_1 + n_0_14_po_2 + n_0_14_po_3 + n_0_14_po_4;
    // relu
    wire [14:0] n_0_14;
    assign n_0_14 = (n_0_14_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_14_sum[14:0]);

    // Layer 0, Neuron 15
    wire signed [15:0] n_0_15_po_0;
    // weight 28: 8'sb00011100
    assign n_0_15_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00011100;

    wire signed [15:0] n_0_15_po_1;
    // weight 22: 8'sb00010110
    assign n_0_15_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00010110;

    wire signed [15:0] n_0_15_po_2;
    // weight 7: 8'sb00000111
    assign n_0_15_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb00000111;

    wire signed [15:0] n_0_15_po_3;
    // weight -28: 8'sb11100100
    assign n_0_15_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11100100;

    wire signed [15:0] n_0_15_po_4;
    // weight 0: 8'sb00000000
    assign n_0_15_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00000000;

    wire signed [15:0] n_0_15_sum;
    assign n_0_15_sum = 8192 + n_0_15_po_0 + n_0_15_po_1 + n_0_15_po_2 + n_0_15_po_3 + n_0_15_po_4;
    // relu
    wire [14:0] n_0_15;
    assign n_0_15 = (n_0_15_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_15_sum[14:0]);

    // Layer 0, Neuron 16
    wire signed [15:0] n_0_16_po_0;
    // weight 5: 8'sb00000101
    assign n_0_16_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000101;

    wire signed [15:0] n_0_16_po_1;
    // weight 12: 8'sb00001100
    assign n_0_16_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00001100;

    wire signed [15:0] n_0_16_po_2;
    // weight -14: 8'sb11110010
    assign n_0_16_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11110010;

    wire signed [15:0] n_0_16_po_3;
    // weight -4: 8'sb11111100
    assign n_0_16_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111100;

    wire signed [15:0] n_0_16_po_4;
    // weight 5: 8'sb00000101
    assign n_0_16_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb00000101;

    wire signed [15:0] n_0_16_sum;
    assign n_0_16_sum = -350 + n_0_16_po_0 + n_0_16_po_1 + n_0_16_po_2 + n_0_16_po_3 + n_0_16_po_4;
    // relu
    wire [14:0] n_0_16;
    assign n_0_16 = (n_0_16_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_16_sum[14:0]);

    // Layer 0, Neuron 17
    wire signed [15:0] n_0_17_po_0;
    // weight -7: 8'sb11111001
    assign n_0_17_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11111001;

    wire signed [15:0] n_0_17_po_1;
    // weight 4: 8'sb00000100
    assign n_0_17_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000100;

    wire signed [15:0] n_0_17_po_2;
    // weight -1: 8'sb11111111
    assign n_0_17_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111111;

    wire signed [15:0] n_0_17_po_3;
    // weight -4: 8'sb11111100
    assign n_0_17_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111100;

    wire signed [15:0] n_0_17_po_4;
    // weight -4: 8'sb11111100
    assign n_0_17_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111100;

    wire signed [15:0] n_0_17_sum;
    assign n_0_17_sum = 3840 + n_0_17_po_0 + n_0_17_po_1 + n_0_17_po_2 + n_0_17_po_3 + n_0_17_po_4;
    // relu
    wire [14:0] n_0_17;
    assign n_0_17 = (n_0_17_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_17_sum[14:0]);

    // Layer 0, Neuron 18
    wire signed [15:0] n_0_18_po_0;
    // weight 0: 8'sb00000000
    assign n_0_18_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb00000000;

    wire signed [15:0] n_0_18_po_1;
    // weight 0: 8'sb00000000
    assign n_0_18_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000000;

    wire signed [15:0] n_0_18_po_2;
    // weight -3: 8'sb11111101
    assign n_0_18_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111101;

    wire signed [15:0] n_0_18_po_3;
    // weight -3: 8'sb11111101
    assign n_0_18_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111101;

    wire signed [15:0] n_0_18_po_4;
    // weight -3: 8'sb11111101
    assign n_0_18_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111101;

    wire signed [15:0] n_0_18_sum;
    assign n_0_18_sum = 2830 + n_0_18_po_0 + n_0_18_po_1 + n_0_18_po_2 + n_0_18_po_3 + n_0_18_po_4;
    // relu
    wire [14:0] n_0_18;
    assign n_0_18 = (n_0_18_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_18_sum[14:0]);

    // Layer 0, Neuron 19
    wire signed [15:0] n_0_19_po_0;
    // weight -1: 8'sb11111111
    assign n_0_19_po_0 = $signed({1'b0, inp[7:0]}) * 8'sb11111111;

    wire signed [15:0] n_0_19_po_1;
    // weight 0: 8'sb00000000
    assign n_0_19_po_1 = $signed({1'b0, inp[15:8]}) * 8'sb00000000;

    wire signed [15:0] n_0_19_po_2;
    // weight -2: 8'sb11111110
    assign n_0_19_po_2 = $signed({1'b0, inp[23:16]}) * 8'sb11111110;

    wire signed [15:0] n_0_19_po_3;
    // weight -2: 8'sb11111110
    assign n_0_19_po_3 = $signed({1'b0, inp[31:24]}) * 8'sb11111110;

    wire signed [15:0] n_0_19_po_4;
    // weight -2: 8'sb11111110
    assign n_0_19_po_4 = $signed({1'b0, inp[39:32]}) * 8'sb11111110;

    wire signed [15:0] n_0_19_sum;
    assign n_0_19_sum = 478 + n_0_19_po_0 + n_0_19_po_1 + n_0_19_po_2 + n_0_19_po_3 + n_0_19_po_4;
    // relu
    wire [14:0] n_0_19;
    assign n_0_19 = (n_0_19_sum<0) ? $unsigned({15{1'b0}}) : $unsigned(n_0_19_sum[14:0]);

    // Layer 1, Neuron 0
    wire signed [22:0] n_1_0_po_0;
    // weight 5: 8'sb00000101
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * 8'sb00000101;

    wire signed [22:0] n_1_0_po_1;
    // weight 16: 8'sb00010000
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * 8'sb00010000;

    wire signed [22:0] n_1_0_po_2;
    // weight 10: 8'sb00001010
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * 8'sb00001010;

    wire signed [22:0] n_1_0_po_3;
    // weight 0: 8'sb00000000
    assign n_1_0_po_3 = $signed({1'b0, n_0_3}) * 8'sb00000000;

    wire signed [22:0] n_1_0_po_4;
    // weight -8: 8'sb11111000
    assign n_1_0_po_4 = $signed({1'b0, n_0_4}) * 8'sb11111000;

    wire signed [22:0] n_1_0_po_5;
    // weight 0: 8'sb00000000
    assign n_1_0_po_5 = $signed({1'b0, n_0_5}) * 8'sb00000000;

    wire signed [22:0] n_1_0_po_6;
    // weight 10: 8'sb00001010
    assign n_1_0_po_6 = $signed({1'b0, n_0_6}) * 8'sb00001010;

    wire signed [22:0] n_1_0_po_7;
    // weight 30: 8'sb00011110
    assign n_1_0_po_7 = $signed({1'b0, n_0_7}) * 8'sb00011110;

    wire signed [22:0] n_1_0_po_8;
    // weight -15: 8'sb11110001
    assign n_1_0_po_8 = $signed({1'b0, n_0_8}) * 8'sb11110001;

    wire signed [22:0] n_1_0_po_9;
    // weight 0: 8'sb00000000
    assign n_1_0_po_9 = $signed({1'b0, n_0_9}) * 8'sb00000000;

    wire signed [22:0] n_1_0_po_10;
    // weight -16: 8'sb11110000
    assign n_1_0_po_10 = $signed({1'b0, n_0_10}) * 8'sb11110000;

    wire signed [22:0] n_1_0_po_11;
    // weight -14: 8'sb11110010
    assign n_1_0_po_11 = $signed({1'b0, n_0_11}) * 8'sb11110010;

    wire signed [22:0] n_1_0_po_12;
    // weight -16: 8'sb11110000
    assign n_1_0_po_12 = $signed({1'b0, n_0_12}) * 8'sb11110000;

    wire signed [22:0] n_1_0_po_13;
    // weight 20: 8'sb00010100
    assign n_1_0_po_13 = $signed({1'b0, n_0_13}) * 8'sb00010100;

    wire signed [22:0] n_1_0_po_14;
    // weight 0: 8'sb00000000
    assign n_1_0_po_14 = $signed({1'b0, n_0_14}) * 8'sb00000000;

    wire signed [22:0] n_1_0_po_15;
    // weight 31: 8'sb00011111
    assign n_1_0_po_15 = $signed({1'b0, n_0_15}) * 8'sb00011111;

    wire signed [22:0] n_1_0_po_16;
    // weight -19: 8'sb11101101
    assign n_1_0_po_16 = $signed({1'b0, n_0_16}) * 8'sb11101101;

    wire signed [22:0] n_1_0_po_17;
    // weight -24: 8'sb11101000
    assign n_1_0_po_17 = $signed({1'b0, n_0_17}) * 8'sb11101000;

    wire signed [22:0] n_1_0_po_18;
    // weight 0: 8'sb00000000
    assign n_1_0_po_18 = $signed({1'b0, n_0_18}) * 8'sb00000000;

    wire signed [22:0] n_1_0_po_19;
    // weight 0: 8'sb00000000
    assign n_1_0_po_19 = $signed({1'b0, n_0_19}) * 8'sb00000000;

    wire signed [23:0] n_1_0_sum;
    assign n_1_0_sum = -256537 + n_1_0_po_0 + n_1_0_po_1 + n_1_0_po_2 + n_1_0_po_3 + n_1_0_po_4 + n_1_0_po_5 + n_1_0_po_6 + n_1_0_po_7 + n_1_0_po_8 + n_1_0_po_9 + n_1_0_po_10 + n_1_0_po_11 + n_1_0_po_12 + n_1_0_po_13 + n_1_0_po_14 + n_1_0_po_15 + n_1_0_po_16 + n_1_0_po_17 + n_1_0_po_18 + n_1_0_po_19;
    // relu
    wire [22:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_0_sum[22:0]);

    // Layer 1, Neuron 1
    wire signed [22:0] n_1_1_po_0;
    // weight 23: 8'sb00010111
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * 8'sb00010111;

    wire signed [22:0] n_1_1_po_1;
    // weight 16: 8'sb00010000
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * 8'sb00010000;

    wire signed [22:0] n_1_1_po_2;
    // weight -88: 8'sb10101000
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * 8'sb10101000;

    wire signed [22:0] n_1_1_po_3;
    // weight 15: 8'sb00001111
    assign n_1_1_po_3 = $signed({1'b0, n_0_3}) * 8'sb00001111;

    wire signed [22:0] n_1_1_po_4;
    // weight 0: 8'sb00000000
    assign n_1_1_po_4 = $signed({1'b0, n_0_4}) * 8'sb00000000;

    wire signed [22:0] n_1_1_po_5;
    // weight 15: 8'sb00001111
    assign n_1_1_po_5 = $signed({1'b0, n_0_5}) * 8'sb00001111;

    wire signed [22:0] n_1_1_po_6;
    // weight -25: 8'sb11100111
    assign n_1_1_po_6 = $signed({1'b0, n_0_6}) * 8'sb11100111;

    wire signed [22:0] n_1_1_po_7;
    // weight -11: 8'sb11110101
    assign n_1_1_po_7 = $signed({1'b0, n_0_7}) * 8'sb11110101;

    wire signed [22:0] n_1_1_po_8;
    // weight -8: 8'sb11111000
    assign n_1_1_po_8 = $signed({1'b0, n_0_8}) * 8'sb11111000;

    wire signed [22:0] n_1_1_po_9;
    // weight -1: 8'sb11111111
    assign n_1_1_po_9 = $signed({1'b0, n_0_9}) * 8'sb11111111;

    wire signed [22:0] n_1_1_po_10;
    // weight -13: 8'sb11110011
    assign n_1_1_po_10 = $signed({1'b0, n_0_10}) * 8'sb11110011;

    wire signed [22:0] n_1_1_po_11;
    // weight 33: 8'sb00100001
    assign n_1_1_po_11 = $signed({1'b0, n_0_11}) * 8'sb00100001;

    wire signed [22:0] n_1_1_po_12;
    // weight -8: 8'sb11111000
    assign n_1_1_po_12 = $signed({1'b0, n_0_12}) * 8'sb11111000;

    wire signed [22:0] n_1_1_po_13;
    // weight -20: 8'sb11101100
    assign n_1_1_po_13 = $signed({1'b0, n_0_13}) * 8'sb11101100;

    wire signed [22:0] n_1_1_po_14;
    // weight 0: 8'sb00000000
    assign n_1_1_po_14 = $signed({1'b0, n_0_14}) * 8'sb00000000;

    wire signed [22:0] n_1_1_po_15;
    // weight 8: 8'sb00001000
    assign n_1_1_po_15 = $signed({1'b0, n_0_15}) * 8'sb00001000;

    wire signed [22:0] n_1_1_po_16;
    // weight 0: 8'sb00000000
    assign n_1_1_po_16 = $signed({1'b0, n_0_16}) * 8'sb00000000;

    wire signed [22:0] n_1_1_po_17;
    // weight 0: 8'sb00000000
    assign n_1_1_po_17 = $signed({1'b0, n_0_17}) * 8'sb00000000;

    wire signed [22:0] n_1_1_po_18;
    // weight 2: 8'sb00000010
    assign n_1_1_po_18 = $signed({1'b0, n_0_18}) * 8'sb00000010;

    wire signed [22:0] n_1_1_po_19;
    // weight 0: 8'sb00000000
    assign n_1_1_po_19 = $signed({1'b0, n_0_19}) * 8'sb00000000;

    wire signed [23:0] n_1_1_sum;
    assign n_1_1_sum = 43699 + n_1_1_po_0 + n_1_1_po_1 + n_1_1_po_2 + n_1_1_po_3 + n_1_1_po_4 + n_1_1_po_5 + n_1_1_po_6 + n_1_1_po_7 + n_1_1_po_8 + n_1_1_po_9 + n_1_1_po_10 + n_1_1_po_11 + n_1_1_po_12 + n_1_1_po_13 + n_1_1_po_14 + n_1_1_po_15 + n_1_1_po_16 + n_1_1_po_17 + n_1_1_po_18 + n_1_1_po_19;
    // relu
    wire [22:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_1_sum[22:0]);

    // Layer 1, Neuron 2
    wire signed [22:0] n_1_2_po_0;
    // weight 11: 8'sb00001011
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * 8'sb00001011;

    wire signed [22:0] n_1_2_po_1;
    // weight 23: 8'sb00010111
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * 8'sb00010111;

    wire signed [22:0] n_1_2_po_2;
    // weight -22: 8'sb11101010
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * 8'sb11101010;

    wire signed [22:0] n_1_2_po_3;
    // weight -18: 8'sb11101110
    assign n_1_2_po_3 = $signed({1'b0, n_0_3}) * 8'sb11101110;

    wire signed [22:0] n_1_2_po_4;
    // weight 27: 8'sb00011011
    assign n_1_2_po_4 = $signed({1'b0, n_0_4}) * 8'sb00011011;

    wire signed [22:0] n_1_2_po_5;
    // weight -17: 8'sb11101111
    assign n_1_2_po_5 = $signed({1'b0, n_0_5}) * 8'sb11101111;

    wire signed [22:0] n_1_2_po_6;
    // weight 20: 8'sb00010100
    assign n_1_2_po_6 = $signed({1'b0, n_0_6}) * 8'sb00010100;

    wire signed [22:0] n_1_2_po_7;
    // weight -22: 8'sb11101010
    assign n_1_2_po_7 = $signed({1'b0, n_0_7}) * 8'sb11101010;

    wire signed [22:0] n_1_2_po_8;
    // weight 24: 8'sb00011000
    assign n_1_2_po_8 = $signed({1'b0, n_0_8}) * 8'sb00011000;

    wire signed [22:0] n_1_2_po_9;
    // weight 0: 8'sb00000000
    assign n_1_2_po_9 = $signed({1'b0, n_0_9}) * 8'sb00000000;

    wire signed [22:0] n_1_2_po_10;
    // weight -5: 8'sb11111011
    assign n_1_2_po_10 = $signed({1'b0, n_0_10}) * 8'sb11111011;

    wire signed [22:0] n_1_2_po_11;
    // weight -7: 8'sb11111001
    assign n_1_2_po_11 = $signed({1'b0, n_0_11}) * 8'sb11111001;

    wire signed [22:0] n_1_2_po_12;
    // weight -18: 8'sb11101110
    assign n_1_2_po_12 = $signed({1'b0, n_0_12}) * 8'sb11101110;

    wire signed [22:0] n_1_2_po_13;
    // weight -6: 8'sb11111010
    assign n_1_2_po_13 = $signed({1'b0, n_0_13}) * 8'sb11111010;

    wire signed [22:0] n_1_2_po_14;
    // weight 0: 8'sb00000000
    assign n_1_2_po_14 = $signed({1'b0, n_0_14}) * 8'sb00000000;

    wire signed [22:0] n_1_2_po_15;
    // weight 1: 8'sb00000001
    assign n_1_2_po_15 = $signed({1'b0, n_0_15}) * 8'sb00000001;

    wire signed [22:0] n_1_2_po_16;
    // weight 22: 8'sb00010110
    assign n_1_2_po_16 = $signed({1'b0, n_0_16}) * 8'sb00010110;

    wire signed [22:0] n_1_2_po_17;
    // weight 20: 8'sb00010100
    assign n_1_2_po_17 = $signed({1'b0, n_0_17}) * 8'sb00010100;

    wire signed [22:0] n_1_2_po_18;
    // weight 0: 8'sb00000000
    assign n_1_2_po_18 = $signed({1'b0, n_0_18}) * 8'sb00000000;

    wire signed [22:0] n_1_2_po_19;
    // weight 0: 8'sb00000000
    assign n_1_2_po_19 = $signed({1'b0, n_0_19}) * 8'sb00000000;

    wire signed [23:0] n_1_2_sum;
    assign n_1_2_sum = 213949 + n_1_2_po_0 + n_1_2_po_1 + n_1_2_po_2 + n_1_2_po_3 + n_1_2_po_4 + n_1_2_po_5 + n_1_2_po_6 + n_1_2_po_7 + n_1_2_po_8 + n_1_2_po_9 + n_1_2_po_10 + n_1_2_po_11 + n_1_2_po_12 + n_1_2_po_13 + n_1_2_po_14 + n_1_2_po_15 + n_1_2_po_16 + n_1_2_po_17 + n_1_2_po_18 + n_1_2_po_19;
    // relu
    wire [22:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({23{1'b0}}) : $unsigned(n_1_2_sum[22:0]);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [23:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign cmp_0_0 = (n_1_0 >= n_1_1);
    assign argmax_val_0_0 = (cmp_0_0) ? n_1_0 : n_1_1;
    assign argmax_idx_0_0 = (cmp_0_0) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [23:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign cmp_1_0 = (argmax_val_0_0 >= n_1_2);
    assign argmax_val_1_0 = (cmp_1_0) ? argmax_val_0_0 : n_1_2;
    assign argmax_idx_1_0 = (cmp_1_0) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;
endmodule
