`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:40:36
// Design Name: 
// Module Name: dwc_unit
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
module dwc_unit#(
    parameter K = 3,
    parameter DATA_W = 8,
    parameter PROD_W = 16,
    parameter PSUM_W = 18
)(
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    
    // Six rows of input feature map
    input  wire signed [DATA_W-1:0] buffer0, 
    input  wire signed [DATA_W-1:0] buffer1, 
    input  wire signed [DATA_W-1:0] buffer2, 
    input  wire signed [DATA_W-1:0] buffer3, 
    input  wire signed [DATA_W-1:0] buffer4, 
    input  wire signed [DATA_W-1:0] buffer5,      

    // Weights (Packed: {w2, w1, w0})
    input  wire [K*DATA_W-1:0] w_col0,
    input  wire [K*DATA_W-1:0] w_col1,
    input  wire [K*DATA_W-1:0] w_col2,

    // Outputs (Wires driven by internal packed DSP registers)
    output wire signed [31:0] out_sum0,
    output wire signed [31:0] out_sum1,
    output wire signed [31:0] out_sum2,
    output wire signed [31:0] out_sum3,
    
    output reg out_valid0,
    output reg out_valid1,
    output reg out_valid2,
    output reg out_valid3
);

    // ============================================================
    // 1. Weight Unpacking
    // ============================================================
    wire signed [DATA_W-1:0] w0_0 = w_col0[7:0];   wire signed [DATA_W-1:0] w0_1 = w_col0[15:8];  wire signed [DATA_W-1:0] w0_2 = w_col0[23:16];
    wire signed [DATA_W-1:0] w1_0 = w_col1[7:0];   wire signed [DATA_W-1:0] w1_1 = w_col1[15:8];  wire signed [DATA_W-1:0] w1_2 = w_col1[23:16];
    wire signed [DATA_W-1:0] w2_0 = w_col2[7:0];   wire signed [DATA_W-1:0] w2_1 = w_col2[15:8];  wire signed [DATA_W-1:0] w2_2 = w_col2[23:16];

    // ============================================================
    // 2. Shared Shift Registers (Line Buffer Slice)
    // ============================================================
    reg signed [DATA_W-1:0] b0_d1, b0_d2;
    reg signed [DATA_W-1:0] b1_d1, b1_d2;
    reg signed [DATA_W-1:0] b2_d1, b2_d2;
    reg signed [DATA_W-1:0] b3_d1, b3_d2;
    reg signed [DATA_W-1:0] b4_d1, b4_d2;
    reg signed [DATA_W-1:0] b5_d1, b5_d2;
    
    reg v_d1, v_d2; 

    always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    b0_d1 <= 0; b0_d2 <= 0;
    b1_d1 <= 0; b1_d2 <= 0;
    b2_d1 <= 0; b2_d2 <= 0;
    b3_d1 <= 0; b3_d2 <= 0;
    b4_d1 <= 0; b4_d2 <= 0;
    b5_d1 <= 0; b5_d2 <= 0;
    v_d1  <= 1'b0;
    v_d2  <= 1'b0;
  end else begin
    if (in_valid) begin
      b0_d1 <= buffer0; b0_d2 <= b0_d1;
      b1_d1 <= buffer1; b1_d2 <= b1_d1;
      b2_d1 <= buffer2; b2_d2 <= b2_d1;
      b3_d1 <= buffer3; b3_d2 <= b3_d1;
      b4_d1 <= buffer4; b4_d2 <= b4_d1;
      b5_d1 <= buffer5; b5_d2 <= b5_d1;

      v_d1 <= 1'b1;
      v_d2 <= v_d1;   
    end else begin
      v_d1 <= 1'b0;
      v_d2 <= 1'b0;
    end
  end
end

    
    (* use_dsp = "yes" *) reg signed [47:0] acc_pack_01; // Stores Row 0 & Row 1
    (* use_dsp = "yes" *) reg signed [47:0] acc_pack_23; // Stores Row 2 & Row 3

    // Packing Function: Concatenates two 8-bit inputs into one 27-bit input
    // Layout: {High_Row (8b), 11b_Zero_Gap, Low_Row (8b)}
    function signed [26:0] pack;
        input signed [7:0] high;
        input signed [7:0] low;
        begin
            pack = {high, 11'd0, low}; 
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_pack_01 <= 0;
            acc_pack_23 <= 0;
            out_valid0 <= 0; out_valid1 <= 0; out_valid2 <= 0; out_valid3 <= 0;
        end else if (v_d2) begin
            
            // --- Pack Row 0 (Low) and Row 1 (High) ---
            // Row 0 inputs: buffer0..2
            // Row 1 inputs: buffer1..3
            // Weights are shared (w2_x, w1_x, w0_x)
            acc_pack_01 <= 
                (pack(buffer1, buffer0) * w2_0) + (pack(buffer2, buffer1) * w2_1) + (pack(buffer3, buffer2) * w2_2) +
                (pack(b1_d1,   b0_d1)   * w1_0) + (pack(b2_d1,   b1_d1)   * w1_1) + (pack(b3_d1,   b2_d1)   * w1_2) +
                (pack(b1_d2,   b0_d2)   * w0_0) + (pack(b2_d2,   b1_d2)   * w0_1) + (pack(b3_d2,   b2_d2)   * w0_2);

            // --- Pack Row 2 (Low) and Row 3 (High) ---
            acc_pack_23 <= 
                (pack(buffer3, buffer2) * w2_0) + (pack(buffer4, buffer3) * w2_1) + (pack(buffer5, buffer4) * w2_2) +
                (pack(b3_d1,   b2_d1)   * w1_0) + (pack(b4_d1,   b3_d1)   * w1_1) + (pack(b5_d1,   b4_d1)   * w1_2) +
                (pack(b3_d2,   b2_d2)   * w0_0) + (pack(b4_d2,   b3_d2)   * w0_1) + (pack(b5_d2,   b4_d2)   * w0_2);

            out_valid0 <= 1'b1; out_valid1 <= 1'b1;
            out_valid2 <= 1'b1; out_valid3 <= 1'b1;
        end else begin
            out_valid0 <= 1'b0; out_valid1 <= 1'b0;
            out_valid2 <= 1'b0; out_valid3 <= 1'b0;
        end
    end

    assign out_sum0 = {{13{acc_pack_01[18]}}, acc_pack_01[18:0]};
    
    // Row 1: High 29 bits of acc_pack_01 (bits 47:19) -> Sign Extended
    assign out_sum1 = {{3{acc_pack_01[47]}},  acc_pack_01[47:19]};

    // Row 2: Low 19 bits of acc_pack_23 -> Sign Extended
    assign out_sum2 = {{13{acc_pack_23[18]}}, acc_pack_23[18:0]};
    
    // Row 3: High 29 bits of acc_pack_23 (bits 47:19) -> Sign Extended
    assign out_sum3 = {{3{acc_pack_23[47]}},  acc_pack_23[47:19]};

endmodule