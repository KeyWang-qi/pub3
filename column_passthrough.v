`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:39:23
// Design Name: 
// Module Name: column_passthrough
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

// Module Name: column_passthrough
// Description: 直通模块，将列数据传递给DWC（替代line_window_packer）
// Date: 2025-11-22
// Author: Peggy-August
//////////////////////////////////////////////////////////////////////////////////

module column_passthrough #(
  parameter integer TILE_H = 6,
  parameter integer UNIT_NUM = 16,
  parameter integer DATA_W = 8
)(
  input  wire clk,
  input  wire rst_n,
  
  // 输入：1列的TILE_H行数据
  input  wire [UNIT_NUM*TILE_H*DATA_W-1:0] column_data_in,
  input  wire column_valid,
  
  // 输出：直接传递
  output reg  [UNIT_NUM*TILE_H*DATA_W-1:0] column_data_out,
  output reg  out_valid
);

  // 添加一级流水线寄存器（可选，用于时序优化）
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      column_data_out <= {UNIT_NUM*TILE_H*DATA_W{1'b0}};
      out_valid <= 1'b0;
    end else begin
      column_data_out <= column_data_in;
      out_valid <= column_valid;
    end
  end

endmodule