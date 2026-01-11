`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:40:01
// Design Name: 
// Module Name: dwc_pu
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
module dwc_pu #(
    parameter UNIT_NUM = 16,
    parameter K = 3,
    parameter DATA_W = 8,
    parameter PROD_W = 16,
    parameter PSUM_W = 18,
    parameter TILE_H = 6,
    parameter integer FAST_SIM_NO_MAC = 0,
    parameter integer FAST_PIPE_LAT = 3
)(

    input wire clk,
    input wire rst_n,
    input wire in_valid,

    input wire [UNIT_NUM*TILE_H*DATA_W-1:0] column_data,

    // ===== ?tap stream =====
    input wire        w_valid,
    input wire [3:0]  w_idx,
    input wire [127:0] w_data,

    output wire [UNIT_NUM*4*32-1:0] out_sums,
    output wire [UNIT_NUM*4-1:0] out_valids
);

    // 9tap?128bit
    reg [127:0] tap_reg [0:8];

    integer t;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (t = 0; t < 9; t = t + 1) begin
                tap_reg[t] <= 128'd0;
            end
        end else if (w_valid) begin
            tap_reg[w_idx] <= w_data;
        end
    end

    // =========  9*128bit  weights =========
    // ? lane(i)w_col0 = {tap2[i],tap1[i],tap0[i]} (??dwc_unit)
    //                 w_col1 = {tap5[i],tap4[i],tap3[i]}
    //                 w_col2 = {tap8[i],tap7[i],tap6[i]}
        wire [UNIT_NUM*3*K*DATA_W-1:0] weights_repacked;

    // Real outputs (from dwc_unit array)
    wire [UNIT_NUM*4*32-1:0] out_sums_real;
    wire [UNIT_NUM*4-1:0]    out_valids_real;

    // FAST mode: simple valid pipeline, sums forced to 0 (keeps downstream dataflow timing)
    reg [FAST_PIPE_LAT-1:0] vpipe;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) vpipe <= {FAST_PIPE_LAT{1'b0}};
      else        vpipe <= {vpipe[FAST_PIPE_LAT-2:0], in_valid};
    end
    wire fast_v = vpipe[FAST_PIPE_LAT-1];

    assign out_sums   = (FAST_SIM_NO_MAC != 0) ? {UNIT_NUM*4*32{1'b0}} : out_sums_real;
    assign out_valids = (FAST_SIM_NO_MAC != 0) ? {UNIT_NUM*4{fast_v}}  : out_valids_real;



    genvar gi;
    generate
      for (gi = 0; gi < UNIT_NUM; gi = gi + 1) begin : REPACK
        // ?lane?8bit
        wire [7:0] w0 = tap_reg[0][gi*8 +: 8];
        wire [7:0] w1 = tap_reg[1][gi*8 +: 8];
        wire [7:0] w2 = tap_reg[2][gi*8 +: 8];
        wire [7:0] w3 = tap_reg[3][gi*8 +: 8];
        wire [7:0] w4 = tap_reg[4][gi*8 +: 8];
        wire [7:0] w5 = tap_reg[5][gi*8 +: 8];
        wire [7:0] w6 = tap_reg[6][gi*8 +: 8];
        wire [7:0] w7 = tap_reg[7][gi*8 +: 8];
        wire [7:0] w8 = tap_reg[8][gi*8 +: 8];

        // col0/1/2  K*DATA_W = 24bit
        assign weights_repacked[
            gi*3*K*DATA_W + 0*K*DATA_W +: K*DATA_W
        ] = {w2, w1, w0};

        assign weights_repacked[
            gi*3*K*DATA_W + 1*K*DATA_W +: K*DATA_W
        ] = {w5, w4, w3};

        assign weights_repacked[
            gi*3*K*DATA_W + 2*K*DATA_W +: K*DATA_W
        ] = {w8, w7, w6};
      end
    endgenerate

    // ========= dwc_pu =========
    genvar i;
    generate
        for (i = 0; i < UNIT_NUM; i = i + 1) begin: dwc_units

            wire [DATA_W-1:0] row_data [0:TILE_H-1];

            for (genvar r = 0; r < TILE_H; r = r + 1) begin
                assign row_data[r] = column_data[i*TILE_H*DATA_W + r*DATA_W +: DATA_W];
            end

            dwc_unit #(
                .K(K),
                .DATA_W(DATA_W),
                .PROD_W(PROD_W),
                .PSUM_W(PSUM_W)
            ) u_dwc (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(in_valid),
                .buffer0(row_data[0]),
                .buffer1(row_data[1]),
                .buffer2(row_data[2]),
                .buffer3(row_data[3]),
                .buffer4(row_data[4]),
                .buffer5(row_data[5]),
                .w_col0(weights_repacked[i*3*K*DATA_W + 0*K*DATA_W +: K*DATA_W]),
                .w_col1(weights_repacked[i*3*K*DATA_W + 1*K*DATA_W +: K*DATA_W]),
                .w_col2(weights_repacked[i*3*K*DATA_W + 2*K*DATA_W +: K*DATA_W]),
                .out_sum0(out_sums_real[i*4*32 + 0*32 +: 32]),
                .out_sum1(out_sums_real[i*4*32 + 1*32 +: 32]),
                .out_sum2(out_sums_real[i*4*32 + 2*32 +: 32]),
                .out_sum3(out_sums_real[i*4*32 + 3*32 +: 32]),
                .out_valid0(out_valids_real[i*4 + 0]),
                .out_valid1(out_valids_real[i*4 + 1]),
                .out_valid2(out_valids_real[i*4 + 2]),
                .out_valid3(out_valids_real[i*4 + 3])
            );
        end
    endgenerate
endmodule