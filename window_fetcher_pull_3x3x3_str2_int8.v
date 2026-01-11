`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:33:43
// Design Name: 
// Module Name: window_fetcher_pull_3x3x3_str2_int8
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
module window_fetcher_pull_3x3x3_str2_int8#(
  parameter integer IMG_W    = 224,
  parameter integer IMG_H    = 224,
  parameter integer A_BITS   = 8,   // per-channel signed
  parameter integer STRIDE   = 2,
  parameter integer PAD      = 1,
  parameter         MEM_PATH = "D:/NoC/mycode/mobilenet_acc2/data/img_224x224x3_s8.mem"
)(
  input  wire                       CLK,
  input  wire                       RESET,       // active-low
  input  wire                       start_frame, // pulse to start a new frame
  input  wire                       win_req,     // request one window
  output reg                        win_valid,   // 1-cycle when win_flat valid
  output reg  [27*A_BITS-1:0]       win_flat,    // cin-fastest
  output reg  [15:0]                win_x,       // optional debug (center x)
  output reg  [15:0]                win_y,       // optional debug (center y)
  output reg                        frame_done   // pulse at last window
);
  localparam integer OUT_W = IMG_W / STRIDE;
  localparam integer OUT_H = IMG_H / STRIDE;

  // BRAM image
  wire [23:0] rom_dout;
  reg         rom_en;
  reg [$clog2(IMG_W*IMG_H)-1:0] rom_addr;

  img_bram_rom #(
    .IMG_W(IMG_W), .IMG_H(IMG_H), .MEM_PATH(MEM_PATH)
  ) u_img (
    .CLK(CLK), .EN(rom_en), .ADDR(rom_addr), .DOUT(rom_dout)
  );

  // Window coordinate (wx, wy)
  reg [15:0] wx, wy;
  reg        running;

  // 9-pixel buffer
  reg [23:0] pix9 [0:8];
  reg [3:0]  idx;         // 0..8
  reg        prev_inb;    // in-bounds for previous issued address
  reg [15:0] x0, y0;      // top-left of 3x3

  // States
  localparam S_IDLE   = 3'd0;
  localparam S_ISSUE  = 3'd1;
  localparam S_CAP    = 3'd2;
  localparam S_PACK   = 3'd3;
  localparam S_OUT    = 3'd4;

  reg [2:0] st, st_n;

  // Helpers
  function [A_BITS-1:0] byteR; input [23:0] p; begin byteR = p[23:16]; end endfunction
  function [A_BITS-1:0] byteG; input [23:0] p; begin byteG = p[15:8];  end endfunction
  function [A_BITS-1:0] byteB; input [23:0] p; begin byteB = p[7:0];   end endfunction

  // Loop indices and temporaries (must be declared at block/module scope for Verilog-2001)
  integer k, kh, kw;
  integer cx, cy;               // coordinate temporaries
  integer addr_calc;            // address calc temporary
  reg [27*A_BITS-1:0] wf_tmp;

  // Next-state
  always @* begin
    st_n = st;
    case (st)
      S_IDLE:  st_n = (running && win_req) ? S_ISSUE : S_IDLE;
      S_ISSUE: st_n = S_CAP;
      S_CAP:   st_n = (idx == 4'd8) ? S_PACK : S_ISSUE;
      S_PACK:  st_n = S_OUT;
      S_OUT:   st_n = S_IDLE;
      default: st_n = S_IDLE;
    endcase
  end

  // Sequential
  always @(posedge CLK or negedge RESET) begin
    if (!RESET) begin
      st <= S_IDLE;
      running <= 1'b0;
      wx <= 0; wy <= 0;
      win_valid <= 1'b0; frame_done <= 1'b0;
      win_flat <= {27*A_BITS{1'b0}};
      win_x <= 16'd0; win_y <= 16'd0;
      idx <= 0; rom_en <= 1'b0; 
      rom_addr <= {($clog2(IMG_W*IMG_H)){1'b0}}; // Verilog-2001 friendly zero
      prev_inb <= 1'b0;
      x0 <= 0; y0 <= 0;
      for (k=0; k<9; k=k+1) pix9[k] <= 24'd0;
    end else begin
      st <= st_n;
      win_valid  <= 1'b0;
      frame_done <= 1'b0;
      rom_en     <= 1'b0;

      // Start new frame
      if (start_frame) begin
        running <= 1'b1;
        wx <= 0; wy <= 0;
      end

      case (st)
        S_IDLE: begin
          if (running && win_req) begin
            // compute top-left of 3x3 for current window center
            x0 <= (wx * STRIDE) - PAD;
            y0 <= (wy * STRIDE) - PAD;
            win_x <= (wx * STRIDE) + (PAD); // center x (padded)
            win_y <= (wy * STRIDE) + (PAD);
            idx <= 0;
            prev_inb <= 1'b0; // first CAP will treat as out-of-bound if we didn't issue
          end
        end

        S_ISSUE: begin
          // compute coordinates for idx
          kw = idx % 3;
          kh = idx / 3;
          // (cx,cy) in original image space (no padding)
          cx = $signed(x0) + $signed(kw);
          cy = $signed(y0) + $signed(kh);
          // issue ROM read if in-bounds
          if ((cx >= 0) && (cx < IMG_W) && (cy >= 0) && (cy < IMG_H)) begin
            rom_en    <= 1'b1;
            addr_calc = (cy * IMG_W) + cx;
            rom_addr  <= addr_calc[$clog2(IMG_W*IMG_H)-1:0];
            prev_inb  <= 1'b1;
          end else begin
            prev_inb <= 1'b0; // out-of-bound -> zero
          end
        end

        S_CAP: begin
          // capture previous read (if any), or zero
          pix9[idx] <= prev_inb ? rom_dout : 24'd0;
          if (idx != 4'd8) begin
            idx <= idx + 1'b1;
          end
        end

        S_PACK: begin
          // pack 9 pixels into cin-fastest layout
          wf_tmp = {27*A_BITS{1'b0}};
          for (kh=0; kh<3; kh=kh+1) begin
            for (kw=0; kw<3; kw=kw+1) begin
              k = kh*3 + kw;
              wf_tmp[k*3*A_BITS +: A_BITS]            = byteR(pix9[k]);
              wf_tmp[k*3*A_BITS + A_BITS +: A_BITS]   = byteG(pix9[k]);
              wf_tmp[k*3*A_BITS + 2*A_BITS +: A_BITS] = byteB(pix9[k]);
            end
          end
          win_flat <= wf_tmp;
        end

        S_OUT: begin
          win_valid <= 1'b1;
          // advance window coordinate
          if (wx == (OUT_W-1)) begin
            wx <= 0;
            if (wy == (OUT_H-1)) begin
              wy <= 0;
              frame_done <= 1'b1;
              // Keep running=1 to allow continuous frames if desired; or gate via start_frame
            end else begin
              wy <= wy + 1'b1;
            end
          end else begin
            wx <= wx + 1'b1;
          end
        end
      endcase
    end
  end
  

endmodule