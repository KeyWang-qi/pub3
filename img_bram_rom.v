`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:34:20
// Design Name: 
// Module Name: img_bram_rom
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
module img_bram_rom#(
  parameter integer IMG_W    = 224,
  parameter integer IMG_H    = 224,
  parameter         MEM_PATH = "D:/NoC/mycode/mobilenet_acc3/data/img_224x224x3_s8.mem"
)(
  input  wire               CLK,
  input  wire               EN,
  input  wire [$clog2(IMG_W*IMG_H)-1:0] ADDR,
  output reg  [23:0]        DOUT
);
  localparam integer NPIX = IMG_W * IMG_H;
  (* rom_style="block" *) reg [23:0] mem [0:NPIX-1]; 
  integer i;

  initial begin
    for (i = 0; i < NPIX; i = i + 1) mem[i] = 24'd0; // [FIX] 初始化为 0，阻断 X 传播
    $readmemh(MEM_PATH, mem);
  end

  always @(posedge CLK) begin
    if (EN) DOUT <= mem[ADDR];
  end
endmodule
