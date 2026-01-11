`timescale 1ns / 1ps
`include "mobilenet_defines.vh"  
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/01/07 15:02:56
// Design Name: 
// Module Name: conv1_scheduler
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
module conv1_scheduler #(
  parameter NUM_ROWS = `PE_ROWS,
  parameter NUM_COLS = `PE_COLS,
  parameter A_BITS   = `DATA_W,
  parameter W_BITS   = `WEIGHT_W,
  parameter ACC_BITS = `ACC_W,
  parameter ADDR_W   = 19
)(
  input  wire                       CLK,
  input  wire                       RESET,   // active-low
  input  wire                       start,
  output reg                        done,

  input  wire [ADDR_W-1:0]          w_base_in,

  output reg                        win_req,
  input  wire                       win_valid,
  input  wire [27*A_BITS-1:0]       win_flat,

  output reg                        weight_req,
  input  wire                       weight_grant,
  output reg  [ADDR_W-1:0]          weight_base,
  output reg  [10:0]                weight_count,
  input  wire                       weight_valid,
  input  wire [127:0]               weight_data,
  input  wire                       weight_done,

  output reg                        arr_W_EN,
  output reg  [NUM_COLS*W_BITS-1:0] in_weight_above, // 256b
  output reg  [NUM_ROWS*A_BITS-1:0] active_left,     // 256b
  input  wire [NUM_COLS*ACC_BITS-1:0] out_sum_final, // 1024b

  output reg                        y_valid,
  output reg  [NUM_COLS*ACC_BITS-1:0] y_data,
  output reg                        y_tile_sel
);

  localparam integer PE_LAT = NUM_ROWS - 1;

  reg [27*A_BITS-1:0] win_reg;

  function [A_BITS-1:0] kbyte;
    input [27*A_BITS-1:0] bus;
    input integer k;
    begin
      kbyte = (k >= 0 && k < 27) ? bus[k*A_BITS +:  A_BITS] : 8'd0;
    end
  endfunction

  wire [NUM_ROWS*A_BITS-1:0] vec0;
  genvar gi;
  generate
    for (gi = 0; gi < NUM_ROWS; gi = gi + 1) begin : GEN_VEC
      if (gi < 27) assign vec0[gi*A_BITS +: A_BITS] = kbyte(win_reg, gi);
      else         assign vec0[gi*A_BITS +: A_BITS] = 8'd0;
    end
  endgenerate

  reg [127:0] w_lo, w_hi;
  reg [1:0]   w_cap_cnt;

  reg [5:0] wait_cnt;
  reg [5:0] cap_col;

  reg signed [ACC_BITS-1:0] psum [0:NUM_COLS-1];
  integer i;

  localparam S_IDLE     = 3'd0;
  localparam S_REQ_WIN  = 3'd1;
  localparam S_WAIT_WIN = 3'd2;
  localparam S_REQ_W    = 3'd3;
  localparam S_WAIT_W   = 3'd4;
  localparam S_INJECT   = 3'd5;
  localparam S_WAITLAT  = 3'd6;
  localparam S_CAPTURE  = 3'd7;
  reg [2:0] state;

  always @(posedge CLK or negedge RESET) begin
    if (!RESET) begin
      state <= S_IDLE;
      done <= 1'b0;
      win_req <= 1'b0;
      win_reg <= 0;

      weight_req <= 1'b0;
      weight_base <= 0;
      weight_count <= 0;

      w_lo <= 0; w_hi <= 0; w_cap_cnt <= 0;

      arr_W_EN <= 1'b0;
      in_weight_above <= 0;
      active_left <= 0;

      wait_cnt <= 0;
      cap_col <= 0;

      y_valid <= 1'b0;
      y_data <= 0;
      y_tile_sel <= 1'b0;

      for (i=0;i<NUM_COLS;i=i+1) psum[i] <= 0;
    end else begin
      done <= 1'b0;
      win_req <= 1'b0;
      y_valid <= 1'b0;
      arr_W_EN <= 1'b0;
      active_left <= 0;

      if (weight_valid) begin
        if (w_cap_cnt == 0) w_lo <= weight_data;
        else if (w_cap_cnt == 1) w_hi <= weight_data;
        if (w_cap_cnt != 2) w_cap_cnt <= w_cap_cnt + 1;
      end

      case (state)
        S_IDLE: begin
          if (start) begin
            win_req <= 1'b1;
            state <= S_REQ_WIN;
          end
        end

        S_REQ_WIN: begin
          win_req <= 1'b1;
          state <= S_WAIT_WIN;
        end

        S_WAIT_WIN: begin
          win_req <= !win_valid;
          if (win_valid) begin
            win_reg <= win_flat;
            state <= S_REQ_W;
          end
        end

        S_REQ_W: begin
          weight_req   <= 1'b1;
          weight_base  <= w_base_in;
          weight_count <= 11'd2; // 2 beats -> 32 cols
          w_cap_cnt    <= 0;
          for (i=0;i<NUM_COLS;i=i+1) psum[i] <= 0;
          state <= S_WAIT_W;
        end

        S_WAIT_W: begin
          if (weight_grant && weight_req) weight_req <= 1'b0;
          if (weight_done) begin
            in_weight_above <= {w_hi, w_lo};
            state <= S_INJECT;
          end
        end

        S_INJECT: begin
          active_left <= vec0;
          wait_cnt <= 0;
          cap_col <= 0;
          arr_W_EN <= 1'b1;
          state <= S_WAITLAT;
        end

        S_WAITLAT: begin
          wait_cnt <= wait_cnt + 1;
          if (wait_cnt >= PE_LAT) state <= S_CAPTURE;
        end

        S_CAPTURE: begin
          psum[cap_col] <= out_sum_final[cap_col*ACC_BITS +: ACC_BITS];
          cap_col <= cap_col + 1;
          if (cap_col == NUM_COLS-1) begin
            for (i=0;i<NUM_COLS;i=i+1)
              y_data[i*ACC_BITS +: ACC_BITS] <= (i == cap_col) ? out_sum_final[i*ACC_BITS +: ACC_BITS] :  psum[i];
            y_valid <= 1'b1;
            y_tile_sel <= 1'b0;
            done <= 1'b1;
            state <= S_IDLE;
          end
        end
      endcase
    end
  end
endmodule
