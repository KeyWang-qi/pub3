`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:26:14
// Design Name: 
// Module Name: unified_feature_buffer
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
module unified_feature_buffer #(
    parameter integer ADDR_W = 16,
    parameter integer WIDTH  = 128,
    parameter integer DEPTH  = 65536
)(
  input  wire                  clk,
  input  wire                  rst_n,

  input  wire                  bank_wr_sel,
  input  wire                  bank_rd_sel,

  // 4-lane write
  input  wire [3:0]            wr_en,
  input  wire [4*ADDR_W-1:0]   wr_addr_vec,
  input  wire [4*WIDTH-1:0]    wr_data_vec,

  // 1-lane read
  input  wire                  rd_en,
  input  wire [ADDR_W-1:0]     rd_addr,
  output reg  [WIDTH-1:0]      rd_data,
  output reg                   rd_valid
);

  localparam integer LANES = 4;
  localparam integer SUB_AW = ADDR_W - 2;
  localparam integer SUB_DEPTH = DEPTH / 4;
  localparam integer RD_LAT = 4;

  // ---- unpack write lanes ----
  wire [ADDR_W-1:0] wr_addr [0:LANES-1];
  wire [WIDTH-1:0]  wr_data [0:LANES-1];

  genvar gi;
  generate
    for (gi=0; gi<LANES; gi=gi+1) begin : GEN_UNPACK
      assign wr_addr[gi] = wr_addr_vec[gi*ADDR_W +: ADDR_W];
      assign wr_data[gi] = wr_data_vec[gi*WIDTH  +: WIDTH ];
    end
  endgenerate


wire [1:0] rd_sb        = rd_addr[1:0] ^ rd_addr[3:2];
wire [SUB_AW-1:0] rd_sa = rd_addr[ADDR_W-1:2];

  wire [1:0] wr_sb [0:LANES-1];
  wire [SUB_AW-1:0] wr_sa [0:LANES-1];

  generate
    for (gi=0; gi<LANES; gi=gi+1) begin : GEN_ADDRMAP
assign wr_sb[gi]        = wr_addr[gi][1:0] ^ wr_addr[gi][3:2];
assign wr_sa[gi]        = wr_addr[gi][ADDR_W-1:2];
    end
  endgenerate

  // ---- read data from 4 subbanks for each ping/pong bank ----
  wire [WIDTH-1:0] dout0 [0:LANES-1];
  wire [WIDTH-1:0] dout1 [0:LANES-1];

  // ---- 8 URAM blocks: bank0_sb0..3 + bank1_sb0..3 ----
  genvar b, s;
  generate
    for (b=0; b<2; b=b+1) begin : GEN_PINGPONG
      for (s=0; s<LANES; s=s+1) begin : GEN_SUBBANK

        // write enable: any lane that targets this subbank AND selected ping/pong bank
        
        wire [127:0] doutb_wire;

        
        wire we_this =
          (bank_wr_sel == b[0]) &&
          ( (wr_en[0] && (wr_sb[0]==s[1:0])) ||
            (wr_en[1] && (wr_sb[1]==s[1:0])) ||
            (wr_en[2] && (wr_sb[2]==s[1:0])) ||
            (wr_en[3] && (wr_sb[3]==s[1:0])) );

        reg [SUB_AW-1:0] waddr_sel;
        reg [WIDTH-1:0]  wdata_sel;
        always @(*) begin
          waddr_sel = {SUB_AW{1'b0}};
          wdata_sel = {WIDTH{1'b0}};
          if (wr_en[3] && bank_wr_sel==b[0] && (wr_sb[3]==s[1:0])) begin waddr_sel=wr_sa[3]; wdata_sel=wr_data[3]; end
          if (wr_en[2] && bank_wr_sel==b[0] && (wr_sb[2]==s[1:0])) begin waddr_sel=wr_sa[2]; wdata_sel=wr_data[2]; end
          if (wr_en[1] && bank_wr_sel==b[0] && (wr_sb[1]==s[1:0])) begin waddr_sel=wr_sa[1]; wdata_sel=wr_data[1]; end
          if (wr_en[0] && bank_wr_sel==b[0] && (wr_sb[0]==s[1:0])) begin waddr_sel=wr_sa[0]; wdata_sel=wr_data[0]; end
        end

        // read enable only when this subbank matches rd_addr[1:0] and bank matches
        wire re_this = rd_en && (bank_rd_sel == b[0]) && (rd_sb == s[1:0]);

        xpm_memory_sdpram #(
          .ADDR_WIDTH_A       (SUB_AW),
          .ADDR_WIDTH_B       (SUB_AW),
          .MEMORY_SIZE        (WIDTH * SUB_DEPTH),
          .MEMORY_PRIMITIVE   ("ultra"),
          .WRITE_DATA_WIDTH_A (WIDTH),
          .READ_DATA_WIDTH_B  (WIDTH),
          .BYTE_WRITE_WIDTH_A (WIDTH),

          .READ_LATENCY_B     (RD_LAT),
          .WRITE_MODE_B       ("read_first"),

          .USE_MEM_INIT       (0),
          .ECC_MODE           ("no_ecc"),
          .WAKEUP_TIME        ("disable_sleep"),
          .AUTO_SLEEP_TIME    (0),
          .MESSAGE_CONTROL    (0)
        ) u_mem (
          .clka   (clk),
          .ena    (we_this),
          .wea    (we_this),
          .addra  (waddr_sel),
          .dina   (wdata_sel),

          .clkb   (clk),
          .enb    (re_this),
          .addrb  (rd_sa),
          .doutb  (doutb_wire),

          .regceb (1'b1),
          .sleep  (1'b0)
        );
if (b == 0) begin : GEN_DOUT0
  assign dout0[s] = doutb_wire;
end else begin : GEN_DOUT1
  assign dout1[s] = doutb_wire;
end


      end
    end
  endgenerate

  // ---- rd_valid pipeline (RD_LAT=4) ----
  reg [RD_LAT-1:0] rd_pipe;
  reg [RD_LAT-1:0] bank_pipe;
  reg [2*RD_LAT-1:0] sb_pipe; // store rd_sb too (2 bits each cycle)

  integer k;
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      rd_pipe  <= {RD_LAT{1'b0}};
      bank_pipe<= {RD_LAT{1'b0}};
      sb_pipe  <= {(2*RD_LAT){1'b0}};
      rd_valid <= 1'b0;
      rd_data  <= {WIDTH{1'b0}};
    end else begin
      rd_pipe   <= {rd_pipe[RD_LAT-2:0], rd_en};
      bank_pipe <= {bank_pipe[RD_LAT-2:0], bank_rd_sel};
      sb_pipe   <= {sb_pipe[2*RD_LAT-3:0], rd_sb};

      rd_valid <= rd_pipe[RD_LAT-1];

      if (rd_pipe[RD_LAT-1]) begin
        // select the correct subbank output after latency
        // sb at output stage:
        case (sb_pipe[2*RD_LAT-1 -: 2])
          2'd0: rd_data <= (bank_pipe[RD_LAT-1]==1'b0) ? dout0[0] : dout1[0];
          2'd1: rd_data <= (bank_pipe[RD_LAT-1]==1'b0) ? dout0[1] : dout1[1];
          2'd2: rd_data <= (bank_pipe[RD_LAT-1]==1'b0) ? dout0[2] : dout1[2];
          2'd3: rd_data <= (bank_pipe[RD_LAT-1]==1'b0) ? dout0[3] : dout1[3];
        endcase
      end
    end
  end

endmodule