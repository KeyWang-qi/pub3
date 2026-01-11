`timescale 1ns / 1ps
`include "mobilenet_defines.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:16:55
// Design Name: 
// Module Name: mobilenet_top_28layers
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
module mobilenet_top_28layers #(
  parameter PE_ROWS = `PE_ROWS,
  parameter PE_COLS = `PE_COLS,
  parameter integer START_LAYER_ID = 6'd0,  
  parameter integer MAX_LAYER_ID   = 6'd28   
)(
  input  wire CLK,
  input  wire RESETn,
  input  wire start,
  output reg  done,
  output wire [2:0] fsm_state,
  output wire [5:0] current_layer,
  
  output wire        fc_out_valid,
  output wire [10:0] fc_out_class_idx,
  output wire signed [7:0] fc_out_logit
);

  localparam integer ADDR_W = 32; 

  localparam S_IDLE = 3'd0;
  localparam S_RUN  = 3'd1;
  localparam S_NEXT = 3'd2;
  localparam S_DONE = 3'd3;

  reg [2:0] state, state_n;
  reg [5:0] cur_layer;
  reg       layer_start;
  wire      layer_done;
  
  wire feat_bank_wr_sel;
  wire feat_bank_rd_sel;

  wire        le_fc_out_valid;
  wire [10:0] le_fc_out_class_idx;
  wire signed [7:0] le_fc_out_logit;

  assign fsm_state     = state;
  assign current_layer = (cur_layer > MAX_LAYER_ID) ? MAX_LAYER_ID : cur_layer;

  always @(*) begin
    state_n     = state;
    layer_start = 1'b0;
    done        = 1'b0;

    case (state)
      S_IDLE: begin
        if (start) begin
          state_n     = S_RUN;
          layer_start = 1'b1;
        end
      end

      S_RUN: begin
        if (layer_done)
          state_n = S_NEXT;
      end

      S_NEXT: begin
        if (cur_layer > MAX_LAYER_ID) begin
          state_n = S_DONE;
        end else begin
          state_n     = S_RUN;
          layer_start = 1'b1;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (! start)
          state_n = S_IDLE;
      end

      default: state_n = S_IDLE;
    endcase
  end

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      state   <= S_IDLE;
      cur_layer <= START_LAYER_ID[5:0];   
    end else begin
      state <= state_n;

      if (state == S_IDLE && start) begin
        cur_layer <= START_LAYER_ID[5:0];  
      end else if (state == S_RUN && layer_done) begin
          cur_layer <= cur_layer + 6'd1;
      end
    end
  end

  // ===== layer_config_rom  =====
  wire [19:0] w_base_cur; 
  wire [11:0] b_base_cur;
  wire [2:0]  layer_type_cur;  
  wire [10:0] cin_cur, cout_cur;
  wire [7:0]  img_w_cur, img_h_cur;
  wire [1:0]  stride_cur;

  wire [4:0] rom_layer_addr = (cur_layer > MAX_LAYER_ID) ? MAX_LAYER_ID : cur_layer;

  layer_config_rom u_layer_cfg (
    .id         (rom_layer_addr),
    .w_base     (w_base_cur),
    .b_base     (b_base_cur),
    .layer_type (layer_type_cur),
    .cin        (cin_cur),
    .cout       (cout_cur),
    .img_w      (img_w_cur),
    .img_h      (img_h_cur),
    .stride     (stride_cur)
  );

  // ===== quant LUT =====
  wire signed [15:0] quant_M;
  wire        [5:0]  quant_s;
  wire signed [7:0]  quant_zp;

  quant_params_lut u_quant_lut (
    .layer_sel    (cur_layer[4:0]),
    .mult_scalar  (quant_M),
    .shift_scalar (quant_s),
    .zp_out       (quant_zp)
  );

  // ===== unified weights + arbiter =====
  localparam integer WBUF_ADDR_W = 15; 
  localparam integer WBUF_DEPTH  = 32768;

  wire                      bmg_ena;
  wire [WBUF_ADDR_W-1:0]    bmg_addra;    
  wire [127:0]              bmg_douta;

  wire                      dma_wr_en;
  wire [WBUF_ADDR_W-1:0]    dma_wr_addr;   
  wire [127:0]              dma_wr_data;

  blk_mem_gen_unified #(
    .BUF_ADDR_W (WBUF_ADDR_W), 
    .WIDTH      (128),
    .DEPTH      (WBUF_DEPTH)  
  ) u_unified_wmem (
    .clka        (CLK),
    .ena         (bmg_ena),
    .addra       (bmg_addra),
    .douta       (bmg_douta),
    .dma_wr_en   (dma_wr_en),
    .dma_wr_addr (dma_wr_addr),
    .dma_wr_data (dma_wr_data)
  );

  // [注意]: 这里的地址线 ADDR_W 现在是 32 位，可以容纳 DDR 大地址
  wire         dw_weight_req,   dw_weight_grant,  dw_weight_valid,  dw_weight_done;
  wire [ADDR_W-1:0] dw_weight_base;
  wire [16:0]       dw_weight_count;
  wire [127:0]      dw_weight_data;

  wire         pw_weight_req,   pw_weight_grant,  pw_weight_valid,  pw_weight_done;
  wire [ADDR_W-1:0] pw_weight_base;
  wire [16:0]       pw_weight_count;
  wire [127:0]      pw_weight_data;

  wire              preload_req;
  wire [ADDR_W-1:0] preload_base;
  wire [16:0]       preload_count;
  wire              preload_done;

  wire                   bmg_en;
  wire [WBUF_ADDR_W-1:0] bmg_addr;  
  wire [127:0]           bmg_data;

  assign bmg_ena   = bmg_en;
  assign bmg_addra = bmg_addr;
  assign bmg_data  = bmg_douta;

  weight_loader_arbiter #(
    .ADDR_W      (ADDR_W),      // 32位：用于处理来自 layer_exec 的 DDR 请求
    .BUF_ADDR_W  (WBUF_ADDR_W)  // 16位：用于寻址片上 BRAM
  ) u_weight_arbiter (
    .clk   (CLK),              
    .rst_n (RESETn),             
    .dw_req   (dw_weight_req),
    .dw_base  (dw_weight_base),
    .dw_count (dw_weight_count),
    .dw_grant (dw_weight_grant),
    .dw_valid (dw_weight_valid),
    .dw_data  (dw_weight_data),
    .dw_done  (dw_weight_done),
    .pw_req   (pw_weight_req),
    .pw_base  (pw_weight_base),
    .pw_count (pw_weight_count),
    .pw_grant (pw_weight_grant),
    .pw_valid (pw_weight_valid),
    .pw_data  (pw_weight_data),
    .pw_done  (pw_weight_done),
    .preload_req   (preload_req),
    .preload_base  (preload_base),
    .preload_count (preload_count),
    .preload_done  (preload_done),
    .bmg_en   (bmg_en),
    .bmg_addr (bmg_addr),
    .bmg_data (bmg_data)
  );

  dma_preload_ctrl_sim #(
    .ADDR_W      (ADDR_W),      
    .DATA_W      (128),
    .BUF_ADDR_W  (WBUF_ADDR_W), // 16位：写入片上 BRAM
    .MEM_FILE    ("D:/NoC/mycode/mobilenet_acc3/data/weights/all_weights_128b.mem") 
  ) u_dma_preload_ctrl_sim (
    .clk              (CLK),
    .rst_n            (RESETn),
    .preload_req      (preload_req),   
    .preload_base     (preload_base),  
    .preload_count    (preload_count), 
    .preload_done     (preload_done),  
    .dma_wr_en        (dma_wr_en),
    .dma_wr_addr      (dma_wr_addr),
    .dma_wr_data      (dma_wr_data)
  );

  // ===== unified feature buffer =====
  wire [3:0]          feat_wr_en;
  wire [4*16-1:0]     feat_wr_addr_vec;   // 4 lanes * 16-bit
  wire [4*128-1:0]    feat_wr_data_vec;   // 4 lanes * 128-bit

  wire         feat_rd_en;
  wire [15:0]  feat_rd_addr;
  wire [127:0] feat_rd_data;
  wire         feat_rd_valid;

  unified_feature_buffer #(
    .ADDR_W (16),   
    .DEPTH  (65536), 
    .WIDTH  (128)
  ) u_feature_buf (
    .clk          (CLK),
    .rst_n        (RESETn),
    .bank_wr_sel  (feat_bank_wr_sel),
    .bank_rd_sel  (feat_bank_rd_sel),
    .wr_en        (feat_wr_en),
    .wr_addr_vec  (feat_wr_addr_vec),
    .wr_data_vec  (feat_wr_data_vec),

    .rd_en        (feat_rd_en),
    .rd_addr      (feat_rd_addr),
    .rd_data      (feat_rd_data),
    .rd_valid     (feat_rd_valid)
  );
  // ===== bias ROM =====
  wire [1023:0] bias_vec;
  wire         bias_valid;
  wire [6:0]   bias_block_idx;
  wire         bias_rd_en;

  bias_rom_unified #(
    .LANES      (32),
    .TOTAL_BIAS (11945)
  ) u_bias_rom (
    .clk          (CLK),
    .rst_n        (RESETn),
    .base_addr_in (b_base_cur),
    .block_idx    (bias_block_idx),
    .rd_en        (bias_rd_en),
    .bias_out     (bias_vec),
    .bias_valid   (bias_valid)
  );

  // ===== layer_exec  =====
  layer_exec #(
  .PE_ROWS(PE_ROWS),
  .PE_COLS(PE_COLS),
  
    .ADDR_W                 (ADDR_W), // 32位，确保能发出正确的 DDR 请求
    .FAST_SIM_EN            (1),
    .FAST_PW_COUT_SUBSAMPLE (4),
    .FAST_PW_PX_SUBSAMPLE   (4),
    .PW_WRITE_LIMIT_EN      (0),
    .PW_WRITE_LIMIT_WORDS   (65535)
  ) u_layer_exec (
    .CLK            (CLK),
    .RESETn         (RESETn),
    .start          (layer_start),
    .done           (layer_done),
    .layer_id       (cur_layer),
    .layer_type     (layer_type_cur), 
    .cin            (cin_cur),
    .cout           (cout_cur),
    .img_w          (img_w_cur),
    .img_h          (img_h_cur),
    .stride         (stride_cur),

    .w_base         ({12'd0, w_base_cur}), 
    .b_base         (b_base_cur),

    .quant_M        (quant_M),
    .quant_s        (quant_s),
    .quant_zp       (quant_zp),
    
    .dw_req         (dw_weight_req),
    .dw_base        (dw_weight_base),
    .dw_count       (dw_weight_count),
    .dw_grant       (dw_weight_grant),
    .dw_valid       (dw_weight_valid),
    .dw_data        (dw_weight_data),
    .dw_done        (dw_weight_done),
    .pw_req         (pw_weight_req),
    .pw_base        (pw_weight_base),
    .pw_count       (pw_weight_count),
    .pw_grant       (pw_weight_grant),
    .pw_valid       (pw_weight_valid),
    .pw_data        (pw_weight_data),
    .pw_done        (pw_weight_done),

    .bias_vec       (bias_vec),
    .bias_valid     (bias_valid),
    .bias_block_idx (bias_block_idx),
    .bias_rd_en     (bias_rd_en),

    .feat_wr_en         (feat_wr_en),
    .feat_wr_local_addr_vec (feat_wr_addr_vec),
    .feat_wr_data_vec       (feat_wr_data_vec),

    .feat_rd_en         (feat_rd_en),
    .feat_rd_local_addr (feat_rd_addr),
    .feat_rd_data       (feat_rd_data),
    .feat_rd_valid      (feat_rd_valid),
     
    .fc_out_valid     (le_fc_out_valid),
    .fc_out_class_idx (le_fc_out_class_idx),
    .fc_out_logit     (le_fc_out_logit),
    .feat_bank_wr_sel (feat_bank_wr_sel),
    .feat_bank_rd_sel (feat_bank_rd_sel)
  );

  assign fc_out_valid     = le_fc_out_valid;
  assign fc_out_class_idx = le_fc_out_class_idx;
  assign fc_out_logit     = le_fc_out_logit;
`ifndef SYNTHESIS
  reg [63:0] gcyc;
  reg [63:0] layer_start_cyc;
  always @(posedge CLK or negedge RESETn) begin
    if(!RESETn) gcyc <= 0;
    else        gcyc <= gcyc + 1;
  end

  always @(posedge CLK) begin
    if (layer_start) layer_start_cyc <= gcyc;
    if (layer_done) begin
      $display("[LAYER %0d] dt=%0d cyc (start=%0d done=%0d)",
        cur_layer, (gcyc - layer_start_cyc), layer_start_cyc, gcyc);
    end
  end
`endif

endmodule