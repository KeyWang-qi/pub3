`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:31:07
// Design Name: 
// Module Name: PE_row_single_weight
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

module PE_row_single_weight#(
    parameter NUM_PE = 32  // PE row
)(

    input wire CLK,
    input wire RESET,
    input wire EN,
    input wire W_EN,
    
    // Activation value input (at the beginning of the line)
    input wire signed [7:0] active_left,
    
    // weight input
    input wire signed [NUM_PE*8-1:0] in_weight_above,
    output wire signed [NUM_PE*8-1:0] out_weight_below,
    
    // psum
    input wire signed [NUM_PE*32-1:0] in_sum,
    output wire signed [NUM_PE*32-1:0] out_sum
);

    // Activation value horizontal interconnect
    wire signed [NUM_PE*8-1:0] active_right;

    // PE row
    genvar gi;
    generate
        for (gi = 0; gi < NUM_PE; gi = gi + 1) begin : pe_gen
            if (gi == 0) begin
                // First PE: Receives external activation value
                PE_single_weight PE_unit(
                    .CLK(CLK),
                    .RESET(RESET),
                    .EN(EN),
                    .W_EN(W_EN),
                    .active_left(active_left),
                    .active_right(active_right[7:0]),
                    .in_sum(in_sum[31:0]),
                    .out_sum(out_sum[31:0]),
                    .in_weight_above(in_weight_above[7:0]),
                    .out_weight_below(out_weight_below[7:0])
                );
            end
            else begin
                // Other PE: Receives the activation value from the previous PE
                PE_single_weight PE_unit(
                    .CLK(CLK),
                    .RESET(RESET),
                    .EN(EN),
                    .W_EN(W_EN),
                    .active_left(active_right[(gi-1)*8+7:(gi-1)*8]),
                    .active_right(active_right[gi*8+7:gi*8]),
                    .in_sum(in_sum[gi*32+31:gi*32]),
                    .out_sum(out_sum[gi*32+31:gi*32]),
                    .in_weight_above(in_weight_above[gi*8+7:gi*8]),
                    .out_weight_below(out_weight_below[gi*8+7:gi*8])
                );
            end
        end
    endgenerate
endmodule

