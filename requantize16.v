`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:33:02
// Design Name: 
// Module Name: requantize16
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
module requantize16#(
  parameter integer LANES           = 16,
  parameter integer ACC_BITS        = 32,
  parameter integer OUT_BITS        = 8
)(
  input  wire                      CLK,
  input  wire                      RESET,
  input  wire                      en,
  input  wire [LANES*ACC_BITS-1:0] in_acc,
  input  wire [LANES*32-1:0]       bias_in,

  input  wire signed [15:0]        cfg_mult_scalar, 
  input  wire        [5:0]         cfg_shift_scalar,
  input  wire                      cfg_symmetric,
  input  wire signed [7:0]         cfg_zp_out,

  output reg  [LANES*OUT_BITS-1:0] out_q,
  output reg                       out_valid
);

  // ============================================================
  // 1. 公共逻辑提取 (Shared Logic) 
  // ============================================================
  // 所有 Lane 的移位量都是一样的，没必要在每个 Lane 里算一遍舍入值。
  // 我们只算一次，然后广播给所有 DSP。
  
  reg signed [47:0] common_round_val;
  reg        [5:0]  common_shift_val;
  reg signed [31:0] common_zp_val;
  reg               common_en_d1, common_en_d2;

  always @(posedge CLK or negedge RESET) begin
    if (!RESET) begin
        common_round_val <= 0;
        common_shift_val <= 0;
        common_zp_val    <= 0;
        common_en_d1 <= 0;
        common_en_d2 <= 0;
        out_valid    <= 0;
    end else begin
        if (en) begin
            // 预计算舍入值： 1 << (shift-1)
            // 这将通过 C 端口进入 DSP，节省了大量 LUT 加法器
            if (cfg_shift_scalar == 0) 
                common_round_val <= 48'd0;
            else 
                common_round_val <= 48'sd1 << (cfg_shift_scalar - 1);

            // 流水线对齐
            common_shift_val <= cfg_shift_scalar;
            
            // 预计算零点 (Symetric ? 0 : ZP)
            common_zp_val    <= cfg_symmetric ? 32'sd0 : {{24{cfg_zp_out[7]}}, cfg_zp_out};
        end
        
        // 使能信号流水线
        common_en_d1 <= en;
        common_en_d2 <= common_en_d1;
        out_valid    <= common_en_d2;
    end
  end

  // ============================================================
  // 2. 辅助函数 (Saturation)
  // ============================================================
  function [7:0] sat_s8; input signed [31:0] x;
    begin
      if (x > 32'sd127)       sat_s8 = 8'sd127;
      else if (x < -32'sd128) sat_s8 = -8'sd128;
      else                    sat_s8 = x[7:0];
    end
  endfunction

  // ============================================================
  // 3. 并行 Lane 处理
  // ============================================================
  genvar gi;
  generate
    for (gi=0; gi<LANES; gi=gi+1) begin : G
      
      // --- Stage 0: Input Unpacking ---
      wire signed [ACC_BITS-1:0] acc_i  = in_acc[gi*ACC_BITS + ACC_BITS-1 : gi*ACC_BITS];
      wire signed [31:0]         bias_i = bias_in[gi*32 + 31 : gi*32];

      // --- Stage 0 (Comb): Pre-clipping ---
      wire signed [32:0] raw_sum = $signed(acc_i) + bias_i; 
      reg  signed [26:0] clamped_sum;
      
      always @(*) begin
        if (raw_sum > 33'sd67108863)       clamped_sum = 27'sd67108863;
        else if (raw_sum < -33'sd67108864) clamped_sum = -27'sd67108864;
        else                               clamped_sum = raw_sum[26:0];
      end

      // --- Stage 1: DSP Multiplication + Rounding Add ---
      // 核心优化：利用 DSP 的 P = A*B + C 结构
      // A = clamped_sum (27b)
      // B = mult_scalar (16b)
      // C = common_round_val (48b) -> 这里的加法免费！不用 LUT！
      
      (* use_dsp = "yes" *)
      reg signed [47:0] prod_rounded; 
      
      always @(posedge CLK or negedge RESET) begin
        if (!RESET) prod_rounded <= 0;
        else if (en) begin

            prod_rounded <= (clamped_sum * $signed(cfg_mult_scalar)) + common_round_val;
        end
      end

      // --- Stage 2: Shifting & ZP Add ---
      reg signed [31:0] shift_res;
      
      always @(posedge CLK or negedge RESET) begin
        if (!RESET) shift_res <= 0;
        else if (common_en_d1) begin
             shift_res <= (prod_rounded >>> common_shift_val) + common_zp_val;
        end
      end
      // --- Stage 3: Saturation & Output ---
      always @(posedge CLK or negedge RESET) begin
        if (!RESET) out_q[gi*OUT_BITS +: OUT_BITS] <= 0;
        else if (common_en_d2) begin
             out_q[gi*OUT_BITS +: OUT_BITS] <= sat_s8(shift_res);
        end
      end

    end
  endgenerate
endmodule