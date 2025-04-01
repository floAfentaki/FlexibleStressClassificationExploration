
`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH = 2;
    parameter NUM_INP = 5;
    parameter WIDTH_A = 8;

    localparam period = 200000000;

    reg [WIDTH_A-1:0] temp_inp [0:NUM_INP-1]; // Temporary storage for inputs
    reg [NUM_INP*WIDTH_A-1:0] inp_reg;        // Register to store concatenated input
    wire [NUM_INP*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    assign inp = inp_reg; // Assign register to wire

    top DUT (
        .inp(inp),
        .out(out)
    );

    integer inFile, outFile, i;
    initial begin
        $display($time, " << Starting the Simulation >>");
        inFile = $fopen("./sim/x_test.txt", "r");
        if (inFile == `NULL) begin
            $display($time, " file not found");
            $finish;
        end
        outFile = $fopen("./sim/y_test.txt", "w");
        while (!$feof(inFile)) begin
            for (i = 0; i < NUM_INP; i = i + 1) begin
                $fscanf(inFile, "%d ", temp_inp[i]);
            end
            $fscanf(inFile, "\n");
            // Concatenate inputs into a single vector
            inp_reg = {temp_inp[0], temp_inp[1], temp_inp[2], temp_inp[3], temp_inp[4]};

            #(period)
            $fwrite(outFile, "%d\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end

    // genvar gi;
    // generate
    // for (gi = 0; gi < NUM_A; gi = gi + 1) begin : genbit
    //    assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    // end
    // endgenerate

endmodule

