`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:30:25
// Design Name: 
// Module Name: PE_array_single_weight
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module PE_array_single_weight#(
    parameter NUM_ROWS = 32,  
    parameter NUM_COLS = 32,
    parameter integer FAST_SIM_NO_MAC = 0
)(

    input wire CLK,
    input wire RESET,
    input wire EN,
    input wire W_EN,
    
    // input from left
    input wire signed [NUM_ROWS*8-1:0] active_left,
    
    // weight from top
    input wire signed [NUM_COLS*8-1:0] in_weight_above,
    output wire signed [NUM_COLS*8-1:0] out_weight_final,
    
    // final data
    output wire signed [NUM_COLS*32-1:0] out_sum_final
);

    // inter-word Weighted Connection
    wire signed [NUM_ROWS*NUM_COLS*8-1:0] out_weight_below;
    
    // psum connection
    wire signed [NUM_ROWS*NUM_COLS*32-1:0] out_sum;
    
    // Zero input (the psum in the first row)
    wire signed [NUM_COLS*32-1:0] zero_sum;
    assign zero_sum = {(NUM_COLS*32){1'b0}};

    // pe array
    genvar gi;
    generate
        for (gi = 0; gi < NUM_ROWS; gi = gi + 1) begin : row_gen
            if (gi == 0) begin
                // first row
                PE_row_single_weight #(.NUM_PE(NUM_COLS)) PE_row_unit(
                    .CLK(CLK),
                    .RESET(RESET),
                    .EN(EN),
                    .W_EN(W_EN),
                    .active_left(active_left[7:0]),
                    .in_weight_above(in_weight_above),
                    .out_weight_below(out_weight_below[NUM_COLS*8-1:0]),
                    .in_sum(zero_sum),
                    .out_sum(out_sum[NUM_COLS*32-1:0])
                );
            end
            else begin
                // other row
                PE_row_single_weight #(.NUM_PE(NUM_COLS)) PE_row_unit(
                    .CLK(CLK),
                    .RESET(RESET),
                    .EN(EN),
                    .W_EN(W_EN),
                    .active_left(active_left[gi*8+7:gi*8]),
                    .in_weight_above(out_weight_below[(gi-1)*NUM_COLS*8+NUM_COLS*8-1:(gi-1)*NUM_COLS*8]),
                    .out_weight_below(out_weight_below[gi*NUM_COLS*8+NUM_COLS*8-1:gi*NUM_COLS*8]),
                    .in_sum(out_sum[(gi-1)*NUM_COLS*32+NUM_COLS*32-1:(gi-1)*NUM_COLS*32]),
                    .out_sum(out_sum[gi*NUM_COLS*32+NUM_COLS*32-1:gi*NUM_COLS*32])
                );
            end
        end
    endgenerate

    // output
    wire signed [NUM_COLS*32-1:0] out_sum_final_real;
    wire signed [NUM_COLS*8-1:0]  out_weight_final_real;

    assign out_sum_final_real    = out_sum[NUM_ROWS*NUM_COLS*32-1:(NUM_ROWS-1)*NUM_COLS*32];
    assign out_weight_final_real = out_weight_below[NUM_ROWS*NUM_COLS*8-1:(NUM_ROWS-1)*NUM_COLS*8];

    // FAST_SIM_NO_MAC: keep IO/scheduling, but bypass MAC-heavy PE rows
    assign out_sum_final    = (FAST_SIM_NO_MAC != 0) ? {NUM_COLS*32{1'b0}} : out_sum_final_real;
    assign out_weight_final = (FAST_SIM_NO_MAC != 0) ? {NUM_COLS*8{1'b0}}  : out_weight_final_real;
endmodule