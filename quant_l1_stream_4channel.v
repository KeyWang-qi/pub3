`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:43:20
// Design Name: 
// Module Name: quant_l1_stream_4channel
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
module quant_l1_stream_4channel #(
  parameter integer UNIT_NUM   = 16,
  parameter integer OUT_W_MAX  = 112,
  parameter integer OUT_H_MAX  = 112,
  parameter integer BLOCKS_MAX = 4,
  parameter integer ACC_W      = 32,
  parameter integer OUT_BITS   = 8
)(
  input  wire clk,
  input  wire rst_n,

  // ----------------------------
  // runtime cfg (?)
  // ----------------------------
  input  wire [7:0] cfg_out_w,
  input  wire [7:0] cfg_out_h,
  input  wire [$clog2(BLOCKS_MAX)-1:0] cfg_blocks,

  // ----------------------------
  // input sums + valids
  // ----------------------------
  input  wire [UNIT_NUM*4*ACC_W-1:0] dwc_sums,
  input  wire [UNIT_NUM*4-1:0]       dwc_valids,

  // ? MAX λlayer_exec ?/?
  input  wire [$clog2(OUT_H_MAX)-1:0]    tile_row,
  input  wire [$clog2(OUT_W_MAX)-1:0]    col_index,
  input  wire [$clog2(BLOCKS_MAX)-1:0]   block_idx,

  input  wire [UNIT_NUM*32-1:0]      bias_vec,
  input  wire signed [15:0]          cfg_mult_scalar,
  input  wire [5:0]                  cfg_shift_scalar,
  input  wire                        cfg_symmetric,
  input  wire signed [7:0]           cfg_zp_out,

  output wire                        wr_en0,
  output wire [15:0]                 wr_addr0,
  output wire [UNIT_NUM*OUT_BITS-1:0] wr_data0,

  output wire                        wr_en1,
  output wire [15:0]                 wr_addr1,
  output wire [UNIT_NUM*OUT_BITS-1:0] wr_data1,

  output wire                        wr_en2,
  output wire [15:0]                 wr_addr2,
  output wire [UNIT_NUM*OUT_BITS-1:0] wr_data2,

  output wire                        wr_en3,
  output wire [15:0]                 wr_addr3,
  output wire [UNIT_NUM*OUT_BITS-1:0] wr_data3
);

  // ============================================================
  // Stage 1: 4-stage pipeline regs (??)
  // ============================================================
  reg [UNIT_NUM*4*ACC_W-1:0]      pipe_sums_0, pipe_sums_1, pipe_sums_2, pipe_sums_3;
  reg [UNIT_NUM*4-1:0]            pipe_valids_0, pipe_valids_1, pipe_valids_2, pipe_valids_3;
  reg [UNIT_NUM*32-1:0]           pipe_bias_0, pipe_bias_1, pipe_bias_2, pipe_bias_3;

  reg [$clog2(OUT_H_MAX)-1:0]     pipe_tile_row_0, pipe_tile_row_1, pipe_tile_row_2, pipe_tile_row_3;
  reg [$clog2(OUT_W_MAX)-1:0]     pipe_col_0, pipe_col_1, pipe_col_2, pipe_col_3;
  reg [$clog2(BLOCKS_MAX)-1:0]    pipe_block_0, pipe_block_1, pipe_block_2, pipe_block_3;

  reg                             pipe_valid_0, pipe_valid_1, pipe_valid_2, pipe_valid_3;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pipe_sums_0 <= 0; pipe_sums_1 <= 0; pipe_sums_2 <= 0; pipe_sums_3 <= 0;
      pipe_valids_0 <= 0; pipe_valids_1 <= 0; pipe_valids_2 <= 0; pipe_valids_3 <= 0;
      pipe_bias_0 <= 0; pipe_bias_1 <= 0; pipe_bias_2 <= 0; pipe_bias_3 <= 0;
      pipe_tile_row_0 <= 0; pipe_tile_row_1 <= 0; pipe_tile_row_2 <= 0; pipe_tile_row_3 <= 0;
      pipe_col_0 <= 0; pipe_col_1 <= 0; pipe_col_2 <= 0; pipe_col_3 <= 0;
      pipe_block_0 <= 0; pipe_block_1 <= 0; pipe_block_2 <= 0; pipe_block_3 <= 0;
      pipe_valid_0 <= 0; pipe_valid_1 <= 0; pipe_valid_2 <= 0; pipe_valid_3 <= 0;
    end else begin
      pipe_sums_3   <= pipe_sums_2;
      pipe_sums_2   <= pipe_sums_1;
      pipe_sums_1   <= pipe_sums_0;

      pipe_valids_3 <= pipe_valids_2;
      pipe_valids_2 <= pipe_valids_1;
      pipe_valids_1 <= pipe_valids_0;

      pipe_bias_3   <= pipe_bias_2;
      pipe_bias_2   <= pipe_bias_1;
      pipe_bias_1   <= pipe_bias_0;

      pipe_tile_row_3 <= pipe_tile_row_2;
      pipe_tile_row_2 <= pipe_tile_row_1;
      pipe_tile_row_1 <= pipe_tile_row_0;

      pipe_col_3 <= pipe_col_2;
      pipe_col_2 <= pipe_col_1;
      pipe_col_1 <= pipe_col_0;

      pipe_block_3 <= pipe_block_2;
      pipe_block_2 <= pipe_block_1;
      pipe_block_1 <= pipe_block_0;

      pipe_valid_3 <= pipe_valid_2;
      pipe_valid_2 <= pipe_valid_1;
      pipe_valid_1 <= pipe_valid_0;

      if (|dwc_valids) begin
        pipe_sums_0     <= dwc_sums;
        pipe_valids_0   <= dwc_valids;
        pipe_bias_0     <= bias_vec;
        pipe_tile_row_0 <= tile_row;
        pipe_col_0      <= col_index;
        pipe_block_0    <= block_idx;
        pipe_valid_0    <= 1'b1;
      end else begin
        pipe_valid_0    <= 1'b0;
      end
    end
  end

  // ============================================================
  // Stage 2: extract per-offset lanes (??)
  // ============================================================
  wire [UNIT_NUM*ACC_W-1:0] lane_acc_0, lane_acc_1, lane_acc_2, lane_acc_3;
  wire [UNIT_NUM-1:0]       lane_valids_0, lane_valids_1, lane_valids_2, lane_valids_3;

  genvar gi;
  generate
    for (gi = 0; gi < UNIT_NUM; gi = gi + 1) begin: EXTRACT_OFF0
      assign lane_acc_0[gi*ACC_W +: ACC_W] = pipe_sums_0[(gi*4 + 0)*ACC_W +: ACC_W];
      assign lane_valids_0[gi]            = pipe_valids_0[gi*4 + 0];
    end
    for (gi = 0; gi < UNIT_NUM; gi = gi + 1) begin: EXTRACT_OFF1
      assign lane_acc_1[gi*ACC_W +: ACC_W] = pipe_sums_1[(gi*4 + 1)*ACC_W +: ACC_W];
      assign lane_valids_1[gi]            = pipe_valids_1[gi*4 + 1];
    end
    for (gi = 0; gi < UNIT_NUM; gi = gi + 1) begin: EXTRACT_OFF2
      assign lane_acc_2[gi*ACC_W +: ACC_W] = pipe_sums_2[(gi*4 + 2)*ACC_W +: ACC_W];
      assign lane_valids_2[gi]            = pipe_valids_2[gi*4 + 2];
    end
    for (gi = 0; gi < UNIT_NUM; gi = gi + 1) begin: EXTRACT_OFF3
      assign lane_acc_3[gi*ACC_W +: ACC_W] = pipe_sums_3[(gi*4 + 3)*ACC_W +: ACC_W];
      assign lane_valids_3[gi]            = pipe_valids_3[gi*4 + 3];
    end
  endgenerate

  // Stage 2 regs
  reg [UNIT_NUM*ACC_W-1:0]  stage2_acc_0, stage2_acc_1, stage2_acc_2, stage2_acc_3;
  reg [UNIT_NUM-1:0]        stage2_valids_0, stage2_valids_1, stage2_valids_2, stage2_valids_3;
  reg [UNIT_NUM*32-1:0]     stage2_bias_0, stage2_bias_1, stage2_bias_2, stage2_bias_3;
  reg [$clog2(OUT_H_MAX)-1:0] stage2_tile_row_0, stage2_tile_row_1, stage2_tile_row_2, stage2_tile_row_3;
  reg [$clog2(OUT_W_MAX)-1:0] stage2_col_0, stage2_col_1, stage2_col_2, stage2_col_3;
  reg [$clog2(BLOCKS_MAX)-1:0] stage2_block_0, stage2_block_1, stage2_block_2, stage2_block_3;
  reg                       stage2_valid_0, stage2_valid_1, stage2_valid_2, stage2_valid_3;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage2_acc_0 <= 0; stage2_acc_1 <= 0; stage2_acc_2 <= 0; stage2_acc_3 <= 0;
      stage2_valids_0 <= 0; stage2_valids_1 <= 0; stage2_valids_2 <= 0; stage2_valids_3 <= 0;
      stage2_bias_0 <= 0; stage2_bias_1 <= 0; stage2_bias_2 <= 0; stage2_bias_3 <= 0;
      stage2_tile_row_0 <= 0; stage2_tile_row_1 <= 0; stage2_tile_row_2 <= 0; stage2_tile_row_3 <= 0;
      stage2_col_0 <= 0; stage2_col_1 <= 0; stage2_col_2 <= 0; stage2_col_3 <= 0;
      stage2_block_0 <= 0; stage2_block_1 <= 0; stage2_block_2 <= 0; stage2_block_3 <= 0;
      stage2_valid_0 <= 0; stage2_valid_1 <= 0; stage2_valid_2 <= 0; stage2_valid_3 <= 0;
    end else begin
      stage2_acc_0 <= lane_acc_0;
      stage2_acc_1 <= lane_acc_1;
      stage2_acc_2 <= lane_acc_2;
      stage2_acc_3 <= lane_acc_3;

      stage2_valids_0 <= lane_valids_0;
      stage2_valids_1 <= lane_valids_1;
      stage2_valids_2 <= lane_valids_2;
      stage2_valids_3 <= lane_valids_3;

      stage2_bias_0 <= pipe_bias_0;
      stage2_bias_1 <= pipe_bias_1;
      stage2_bias_2 <= pipe_bias_2;
      stage2_bias_3 <= pipe_bias_3;

      stage2_tile_row_0 <= pipe_tile_row_0;
      stage2_tile_row_1 <= pipe_tile_row_1;
      stage2_tile_row_2 <= pipe_tile_row_2;
      stage2_tile_row_3 <= pipe_tile_row_3;

      stage2_col_0 <= pipe_col_0;
      stage2_col_1 <= pipe_col_1;
      stage2_col_2 <= pipe_col_2;
      stage2_col_3 <= pipe_col_3;

      stage2_block_0 <= pipe_block_0;
      stage2_block_1 <= pipe_block_1;
      stage2_block_2 <= pipe_block_2;
      stage2_block_3 <= pipe_block_3;

      stage2_valid_0 <= pipe_valid_0 && (|lane_valids_0);
      stage2_valid_1 <= pipe_valid_1 && (|lane_valids_1);
      stage2_valid_2 <= pipe_valid_2 && (|lane_valids_2);
      stage2_valid_3 <= pipe_valid_3 && (|lane_valids_3);
    end
  end

  // ============================================================
  // Stage 3: requantize16 (??)
  // ============================================================
  wire [UNIT_NUM*OUT_BITS-1:0] rq_out_q_0, rq_out_q_1, rq_out_q_2, rq_out_q_3;
  wire rq_out_valid_0, rq_out_valid_1, rq_out_valid_2, rq_out_valid_3;

  requantize16 #(.LANES(UNIT_NUM), .ACC_BITS(ACC_W), .OUT_BITS(OUT_BITS)) u_rq_off0 (
    .CLK(clk), .RESET(rst_n), .en(stage2_valid_0),
    .in_acc(stage2_acc_0), .bias_in(stage2_bias_0),
    .cfg_mult_scalar(cfg_mult_scalar), .cfg_shift_scalar(cfg_shift_scalar),
    .cfg_symmetric(cfg_symmetric), .cfg_zp_out(cfg_zp_out),
    .out_q(rq_out_q_0), .out_valid(rq_out_valid_0)
  );
  requantize16 #(.LANES(UNIT_NUM), .ACC_BITS(ACC_W), .OUT_BITS(OUT_BITS)) u_rq_off1 (
    .CLK(clk), .RESET(rst_n), .en(stage2_valid_1),
    .in_acc(stage2_acc_1), .bias_in(stage2_bias_1),
    .cfg_mult_scalar(cfg_mult_scalar), .cfg_shift_scalar(cfg_shift_scalar),
    .cfg_symmetric(cfg_symmetric), .cfg_zp_out(cfg_zp_out),
    .out_q(rq_out_q_1), .out_valid(rq_out_valid_1)
  );
  requantize16 #(.LANES(UNIT_NUM), .ACC_BITS(ACC_W), .OUT_BITS(OUT_BITS)) u_rq_off2 (
    .CLK(clk), .RESET(rst_n), .en(stage2_valid_2),
    .in_acc(stage2_acc_2), .bias_in(stage2_bias_2),
    .cfg_mult_scalar(cfg_mult_scalar), .cfg_shift_scalar(cfg_shift_scalar),
    .cfg_symmetric(cfg_symmetric), .cfg_zp_out(cfg_zp_out),
    .out_q(rq_out_q_2), .out_valid(rq_out_valid_2)
  );
  requantize16 #(.LANES(UNIT_NUM), .ACC_BITS(ACC_W), .OUT_BITS(OUT_BITS)) u_rq_off3 (
    .CLK(clk), .RESET(rst_n), .en(stage2_valid_3),
    .in_acc(stage2_acc_3), .bias_in(stage2_bias_3),
    .cfg_mult_scalar(cfg_mult_scalar), .cfg_shift_scalar(cfg_shift_scalar),
    .cfg_symmetric(cfg_symmetric), .cfg_zp_out(cfg_zp_out),
    .out_q(rq_out_q_3), .out_valid(rq_out_valid_3)
  );

  // ============================================================
  // Stage 4:  tile/col/block ?? runtime cfg
  // ============================================================
  reg [$clog2(OUT_H_MAX)-1:0] stage4_tile_row_0, stage4_tile_row_1, stage4_tile_row_2, stage4_tile_row_3;
  reg [$clog2(OUT_W_MAX)-1:0] stage4_col_0, stage4_col_1, stage4_col_2, stage4_col_3;
  reg [$clog2(BLOCKS_MAX)-1:0] stage4_block_0, stage4_block_1, stage4_block_2, stage4_block_3;
  reg stage4_valid_0, stage4_valid_1, stage4_valid_2, stage4_valid_3;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage4_tile_row_0 <= 0; stage4_tile_row_1 <= 0; stage4_tile_row_2 <= 0; stage4_tile_row_3 <= 0;
      stage4_col_0 <= 0; stage4_col_1 <= 0; stage4_col_2 <= 0; stage4_col_3 <= 0;
      stage4_block_0 <= 0; stage4_block_1 <= 0; stage4_block_2 <= 0; stage4_block_3 <= 0;
      stage4_valid_0 <= 0; stage4_valid_1 <= 0; stage4_valid_2 <= 0; stage4_valid_3 <= 0;
    end else begin
      stage4_tile_row_0 <= stage2_tile_row_0;
      stage4_tile_row_1 <= stage2_tile_row_1;
      stage4_tile_row_2 <= stage2_tile_row_2;
      stage4_tile_row_3 <= stage2_tile_row_3;

      stage4_col_0 <= stage2_col_0;
      stage4_col_1 <= stage2_col_1;
      stage4_col_2 <= stage2_col_2;
      stage4_col_3 <= stage2_col_3;

      stage4_block_0 <= stage2_block_0;
      stage4_block_1 <= stage2_block_1;
      stage4_block_2 <= stage2_block_2;
      stage4_block_3 <= stage2_block_3;

      stage4_valid_0 <= stage2_valid_0;
      stage4_valid_1 <= stage2_valid_1;
      stage4_valid_2 <= stage2_valid_2;
      stage4_valid_3 <= stage2_valid_3;
    end
  end

  // ?? (row*cfg_out_w + col) * cfg_blocks + blk
  function [15:0] calc_local_addr_cfg;
    input [15:0] row;
    input [15:0] col;
    input [15:0] blk;
    input [7:0]  out_w;
    input [15:0]  blocks;
    reg [31:0] tmp;
    begin
      tmp = (row * out_w + col) * blocks + blk;
      calc_local_addr_cfg = tmp[15:0];
    end
  endfunction

  // row offset 0..3
  wire [15:0] row0 = {{(16-$clog2(OUT_H_MAX)){1'b0}}, stage4_tile_row_0} + 16'd0;
  wire [15:0] row1 = {{(16-$clog2(OUT_H_MAX)){1'b0}}, stage4_tile_row_1} + 16'd1;
  wire [15:0] row2 = {{(16-$clog2(OUT_H_MAX)){1'b0}}, stage4_tile_row_2} + 16'd2;
  wire [15:0] row3 = {{(16-$clog2(OUT_H_MAX)){1'b0}}, stage4_tile_row_3} + 16'd3;

  wire [15:0] col0 = {{(16-$clog2(OUT_W_MAX)){1'b0}}, stage4_col_0};
  wire [15:0] col1 = {{(16-$clog2(OUT_W_MAX)){1'b0}}, stage4_col_1};
  wire [15:0] col2 = {{(16-$clog2(OUT_W_MAX)){1'b0}}, stage4_col_2};
  wire [15:0] col3 = {{(16-$clog2(OUT_W_MAX)){1'b0}}, stage4_col_3};

  wire [15:0] blk0 = {{(16-$clog2(BLOCKS_MAX)){1'b0}}, stage4_block_0};
  wire [15:0] blk1 = {{(16-$clog2(BLOCKS_MAX)){1'b0}}, stage4_block_1};
  wire [15:0] blk2 = {{(16-$clog2(BLOCKS_MAX)){1'b0}}, stage4_block_2};
  wire [15:0] blk3 = {{(16-$clog2(BLOCKS_MAX)){1'b0}}, stage4_block_3};

  wire [15:0] calc_addr_0 = calc_local_addr_cfg(row0, col0, blk0, cfg_out_w, cfg_blocks);
  wire [15:0] calc_addr_1 = calc_local_addr_cfg(row1, col1, blk1, cfg_out_w, cfg_blocks);
  wire [15:0] calc_addr_2 = calc_local_addr_cfg(row2, col2, blk2, cfg_out_w, cfg_blocks);
  wire [15:0] calc_addr_3 = calc_local_addr_cfg(row3, col3, blk3, cfg_out_w, cfg_blocks);

// 1. 先声明 wire 信号
  wire inrange0;
  wire inrange1;
  wire inrange2;
  wire inrange3;

  // 2. 再使用 assign 进行赋值
  // bounds check (avoid writing past cfg_out_w/h)
  assign inrange0 = (row0 < {8'd0, cfg_out_h}) && (col0 < {8'd0, cfg_out_w});
  assign inrange1 = (row1 < {8'd0, cfg_out_h}) && (col1 < {8'd0, cfg_out_w});
  assign inrange2 = (row2 < {8'd0, cfg_out_h}) && (col2 < {8'd0, cfg_out_w});
  assign inrange3 = (row3 < {8'd0, cfg_out_h}) && (col3 < {8'd0, cfg_out_w});


  reg wr_en_0_reg, wr_en_1_reg, wr_en_2_reg, wr_en_3_reg;
  reg [15:0] wr_addr_0_reg, wr_addr_1_reg, wr_addr_2_reg, wr_addr_3_reg;
  reg [UNIT_NUM*OUT_BITS-1:0] wr_data_0_reg, wr_data_1_reg, wr_data_2_reg, wr_data_3_reg;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_en_0_reg <= 0; wr_en_1_reg <= 0; wr_en_2_reg <= 0; wr_en_3_reg <= 0;
      wr_addr_0_reg <= 0; wr_addr_1_reg <= 0; wr_addr_2_reg <= 0; wr_addr_3_reg <= 0;
      wr_data_0_reg <= 0; wr_data_1_reg <= 0; wr_data_2_reg <= 0; wr_data_3_reg <= 0;
    end else begin
      wr_en_0_reg <= (rq_out_valid_0 && stage4_valid_0 && inrange0);
      wr_en_1_reg <= (rq_out_valid_1 && stage4_valid_1 && inrange1);
      wr_en_2_reg <= (rq_out_valid_2 && stage4_valid_2 && inrange2);
      wr_en_3_reg <= (rq_out_valid_3 && stage4_valid_3 && inrange3);

      if (rq_out_valid_0 && stage4_valid_0 && inrange0) begin
        wr_addr_0_reg <= calc_addr_0;
        wr_data_0_reg <= rq_out_q_0;
      end
      if (rq_out_valid_1 && stage4_valid_1 && inrange1) begin
        wr_addr_1_reg <= calc_addr_1;
        wr_data_1_reg <= rq_out_q_1;
      end
      if (rq_out_valid_2 && stage4_valid_2 && inrange2) begin
        wr_addr_2_reg <= calc_addr_2;
        wr_data_2_reg <= rq_out_q_2;
      end
      if (rq_out_valid_3 && stage4_valid_3 && inrange3) begin
        wr_addr_3_reg <= calc_addr_3;
        wr_data_3_reg <= rq_out_q_3;
      end
    end
  end

  assign wr_en0   = wr_en_0_reg;
  assign wr_addr0 = wr_addr_0_reg;
  assign wr_data0 = wr_data_0_reg;

  assign wr_en1   = wr_en_1_reg;
  assign wr_addr1 = wr_addr_1_reg;
  assign wr_data1 = wr_data_1_reg;

  assign wr_en2   = wr_en_2_reg;
  assign wr_addr2 = wr_addr_2_reg;
  assign wr_data2 = wr_data_2_reg;

  assign wr_en3   = wr_en_3_reg;
  assign wr_addr3 = wr_addr_3_reg;
  assign wr_data3 = wr_data_3_reg;

endmodule