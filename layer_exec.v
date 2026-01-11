`timescale 1ns / 1ps
`include "mobilenet_defines.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:29:20
// Design Name: 
// Module Name: layer_exec
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
// Description: Per-layer execution unit for MobileNet top-28layers.
module layer_exec #(
  parameter PE_ROWS = `PE_ROWS,
  parameter PE_COLS = `PE_COLS,
  parameter integer ADDR_W = 16,
  parameter integer FAST_SIM_EN            = 0,
  parameter integer FAST_PW_COUT_SUBSAMPLE = 16,
  parameter integer FAST_PW_PX_SUBSAMPLE   = 16,
  parameter integer PW_WRITE_LIMIT_EN      = 1,
  parameter integer PW_WRITE_LIMIT_WORDS   = 2048,
  parameter integer USE_PW32x16            = 1
)(
  input  wire CLK,
  input  wire RESETn,
  input  wire start,
  output reg  done,

  input  wire [5:0] layer_id,
  input  wire [2:0] layer_type,
  input  wire [10:0] cin,
  input  wire [10:0] cout,
  input  wire [7:0]  img_w,
  input  wire [7:0]  img_h,
  input  wire [1:0]  stride,

  input  wire [ADDR_W-1:0] w_base,
  input  wire [11:0]       b_base,

  input  wire signed [15:0] quant_M,
  input  wire        [5:0]  quant_s,
  input  wire signed [7:0]  quant_zp,

  // DW channel
  output wire               dw_req,
  output wire [ADDR_W-1:0]  dw_base,
  output wire [16:0]        dw_count,
  input  wire               dw_grant,
  input  wire               dw_valid,
  input  wire [127:0]       dw_data,
  input  wire               dw_done,

  // PW / CONV channel
  output wire               pw_req,
  output wire [ADDR_W-1:0]  pw_base,
  output wire [16:0]        pw_count,
  input  wire               pw_grant,
  input  wire               pw_valid,
  input  wire [127:0]       pw_data,
  input  wire               pw_done,

  // Bias ROM
  input  wire [1023:0]       bias_vec,
  input  wire               bias_valid,
  output reg  [6:0]         bias_block_idx,
  output reg                bias_rd_en,

  // Feature buffer
  output wire [3:0]         feat_wr_en,
  output wire [4*16-1:0]    feat_wr_local_addr_vec,
  output wire [4*128-1:0]   feat_wr_data_vec,

  output wire               feat_rd_en,
  output wire [15:0]        feat_rd_local_addr,
  input  wire [127:0]       feat_rd_data,
  input  wire               feat_rd_valid,

  // FC out
  output wire               fc_out_valid,
  output wire [10:0]        fc_out_class_idx,
  output wire signed [7:0]  fc_out_logit,

  output reg feat_bank_wr_sel,
  output reg feat_bank_rd_sel
);
  // ============================================================
  // PE
  // ============================================================

  localparam [2:0] TYPE_CONV = 3'd0;
  localparam [2:0] TYPE_DW   = 3'd1;
  localparam [2:0] TYPE_PW   = 3'd2;
  localparam [2:0] TYPE_AP   = 3'd3;
  localparam [2:0] TYPE_FC   = 3'd4;

  // DW generic
  localparam integer DW_UNIT = 16;

  wire is_dw_layer = (layer_type == TYPE_DW);
  wire is_ap_layer = (layer_type == TYPE_AP);
  wire is_fc_layer = (layer_type == TYPE_FC);

  wire [6:0] dw_num_blocks  = (cin + DW_UNIT - 1) / DW_UNIT;
  reg  [6:0] dw_block_idx;
  wire       dw_last_block  = (dw_block_idx == (dw_num_blocks - 1));
  wire       dw_stride2     = (stride == 2'd2);

  reg        dw_block_started;
  wire       dw_block_start;

  // ============================================================
  // 1) Local FSM
  // ============================================================
  localparam LE_IDLE    = 4'd0;
  localparam LE_RUN_L0  = 4'd1;
  localparam LE_RUN_L1D = 4'd2;
  localparam LE_RUN_L2P = 4'd3;
  localparam LE_RUN_AP  = 4'd4;
  localparam LE_RUN_FC  = 4'd5;
  localparam LE_BYPASS  = 4'd6;

  reg [3:0] le_state, le_state_n;

  reg  l0_start;  wire l0_done;
  wire l1_done;
  reg  l2_start;  wire l2_done;
  reg  ap_start;  wire ap_done;
  reg  fc_start;  wire fc_done;

  wire [6:0] fc_bias_block_idx;

  // FSM regs + pingpong bank swap
  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      le_state <= LE_IDLE;
      feat_bank_wr_sel <= 1'b0;
      feat_bank_rd_sel <= 1'b1;
    end else begin
      le_state <= le_state_n;

      if (
        (le_state == LE_RUN_L0  && l0_done) ||
        (le_state == LE_RUN_L1D && l1_done) ||
        (le_state == LE_RUN_L2P && l2_done) ||
        (le_state == LE_RUN_FC  && fc_done)
      ) begin
        feat_bank_wr_sel <= ~feat_bank_wr_sel;
        feat_bank_rd_sel <= ~feat_bank_rd_sel;
      end
    end
  end

  // FSM comb
  always @(*) begin
    le_state_n = le_state;
    done       = 1'b0;

    l0_start   = 1'b0;
    l2_start   = 1'b0;
    ap_start   = 1'b0;
    fc_start   = 1'b0;

    bias_rd_en     = 1'b0;
    bias_block_idx = dw_block_idx;

    case (le_state)
      LE_IDLE: begin
        if (start) begin
          case (layer_type)
            TYPE_CONV: begin
              if (layer_id == 6'd0) begin
                le_state_n = LE_RUN_L0;
                l0_start   = 1'b1;
              end else begin
                le_state_n = LE_BYPASS;
              end
            end
            TYPE_DW: begin
              le_state_n = LE_RUN_L1D;
            end
            TYPE_PW: begin
              le_state_n = LE_RUN_L2P;
              l2_start   = 1'b1;
            end
            TYPE_AP: begin
              le_state_n = LE_RUN_AP;
              ap_start   = 1'b1;
            end
            TYPE_FC: begin
              le_state_n = LE_RUN_FC;
              fc_start   = 1'b1;
            end
            default: begin
              le_state_n = LE_BYPASS;
            end
          endcase
        end
      end

      LE_RUN_L0:  begin if (l0_done) begin le_state_n = LE_IDLE; done = 1'b1; end end
      LE_RUN_L1D: begin
        bias_rd_en     = 1'b1;
        bias_block_idx = dw_block_idx;
        if (l1_done) begin le_state_n = LE_IDLE; done = 1'b1; end
      end
      LE_RUN_L2P: begin if (l2_done) begin le_state_n = LE_IDLE; done = 1'b1; end end
      LE_RUN_AP:  begin if (ap_done) begin le_state_n = LE_IDLE; done = 1'b1; end end
      LE_RUN_FC:  begin
        bias_rd_en     = 1'b1;
        bias_block_idx = fc_bias_block_idx;
        if (fc_done) begin le_state_n = LE_IDLE; done = 1'b1; end
      end

      LE_BYPASS: begin done = 1'b1; le_state_n = LE_IDLE; end
      default:  begin le_state_n = LE_IDLE; end
    endcase
  end
  
// ============================================================
  // 2) Shared PE array (L0/L2)
  // ============================================================
  wire [PE_ROWS*8-1:0]  arr_active_left;    // 32*8 = 256b
  wire [PE_COLS*8-1:0]  arr_weight_above;   // 32*8 = 256b
  wire                  arr_w_en;
  wire [PE_COLS*32-1:0] arr_out_sum;        // 32*32 = 1024b

  (* keep_hierarchy = "yes", dont_touch = "true" *)
  PE_array_single_weight #(
    .NUM_ROWS(PE_ROWS),
    .NUM_COLS(PE_COLS)
  ) u_shared_array (
    .CLK(CLK),
    .RESET(RESETn),
    .EN(1'b1),
    .W_EN(arr_w_en),
    .active_left(arr_active_left),
    .in_weight_above(arr_weight_above),
    .out_weight_final(),
    .out_sum_final(arr_out_sum)
  );

  // L0 (conv1) 32x32 weight view
  wire [PE_ROWS*8-1:0] l0_act;
  wire                 l0_wen;
  wire [255:0]         l0_w32;

  // L2 (PW) 32x32 weight view
  wire [PE_ROWS*8-1:0] l2_act;
  wire                 l2_wen;
  wire [255:0]         l2_w32;

  assign arr_active_left =
    (le_state == LE_RUN_L0)  ? l0_act :
    (le_state == LE_RUN_L2P) ? l2_act :
    {(PE_ROWS*8){1'b0}};

  assign arr_weight_above =
    (le_state == LE_RUN_L0)  ? l0_w32 :
    (le_state == LE_RUN_L2P) ? l2_w32 :
    {(PE_COLS*8){1'b0}};

  assign arr_w_en =
    (le_state == LE_RUN_L0)  ? l0_wen :
    (le_state == LE_RUN_L2P) ? l2_wen :
    1'b0;

  // ============================================================
  // 3) Shared requantize16
  // ============================================================
  wire        l0_y_valid;  wire [1023:0] l0_y_data;
  wire        l2_y_valid;  wire [1023:0] l2_y_data;

reg        q_en;
reg [1023:0] q_in;

always @(posedge CLK) begin
  case (le_state)
    LE_RUN_L0:  begin q_en <= l0_y_valid; q_in <= l0_y_data; end
    LE_RUN_L2P: begin q_en <= l2_y_valid; q_in <= l2_y_data; end
    default:    begin q_en <= 1'b0;       q_in <= 1024'd0;  end
  endcase
end


  wire [255:0] q_out_data;
  wire         q_out_valid;

  requantize16#(.LANES(32)) u_shared_quant (
    .CLK             (CLK),
    .RESET           (RESETn),
    .en              (q_en),
    .in_acc          (q_in),
    .bias_in         (bias_vec),
    .cfg_mult_scalar (quant_M),
    .cfg_shift_scalar(quant_s),
    .cfg_symmetric   (1'b0),
    .cfg_zp_out      (quant_zp),
    .out_q           (q_out_data),
    .out_valid       (q_out_valid)
  );

 // ============================================================
  // 4) Feature buffer mux
  // ============================================================
  wire       l1_feat_rd_en;   wire [15:0] l1_feat_rd_addr;
  wire       l2_feat_rd_en;   wire [15:0] l2_feat_rd_addr;
  wire       ap_feat_rd_en;   wire [15:0] ap_feat_rd_addr;
  wire       fc_feat_rd_en;   wire [15:0] fc_feat_rd_addr;

  wire       l0_feat_wr_en;   wire [15:0] l0_feat_wr_addr; wire [127:0] l0_feat_wr_data;
  wire [3:0] l1_feat_wr_en_vec; wire [4*16-1:0] l1_feat_wr_addr_vec; wire [4*128-1:0] l1_feat_wr_data_vec;
  wire       l2_feat_wr_en;   wire [15:0] l2_feat_wr_addr; wire [127:0] l2_feat_wr_data;
  wire       ap_feat_wr_en;   wire [15:0] ap_feat_wr_addr; wire [127:0] ap_feat_wr_data;
  wire       fc_feat_wr_en;   wire [15:0] fc_feat_wr_addr; wire [127:0] fc_feat_wr_data;

    // Global write mux (4-lane)
  // lane0 is [15:0] / [127:0] in the *_vec buses
  assign feat_wr_en =
    (le_state == LE_RUN_L0)  ? {3'b000, l0_feat_wr_en} :
    (le_state == LE_RUN_L1D) ? l1_feat_wr_en_vec :
    (le_state == LE_RUN_L2P) ? {3'b000, l2_feat_wr_en} :
    (le_state == LE_RUN_AP)  ? {3'b000, ap_feat_wr_en} :
    (le_state == LE_RUN_FC)  ? {3'b000, fc_feat_wr_en} :
                               4'b0000;

  assign feat_wr_local_addr_vec =
    (le_state == LE_RUN_L0)  ? {48'd0, l0_feat_wr_addr} :
    (le_state == LE_RUN_L1D) ? l1_feat_wr_addr_vec :
    (le_state == LE_RUN_L2P) ? {48'd0, l2_feat_wr_addr} :
    (le_state == LE_RUN_AP)  ? {48'd0, ap_feat_wr_addr} :
    (le_state == LE_RUN_FC)  ? {48'd0, fc_feat_wr_addr} :
                               64'd0;

  assign feat_wr_data_vec =
    (le_state == LE_RUN_L0)  ? {384'd0, l0_feat_wr_data} :
    (le_state == LE_RUN_L1D) ? l1_feat_wr_data_vec :
    (le_state == LE_RUN_L2P) ? {384'd0, l2_feat_wr_data} :
    (le_state == LE_RUN_AP)  ? {384'd0, ap_feat_wr_data} :
    (le_state == LE_RUN_FC)  ? {384'd0, fc_feat_wr_data} :
                               512'd0;
assign feat_rd_en =
    (le_state == LE_RUN_L1D) ? l1_feat_rd_en :
    (le_state == LE_RUN_L2P) ? l2_feat_rd_en :
    (le_state == LE_RUN_AP)  ? ap_feat_rd_en :
    (le_state == LE_RUN_FC)  ? fc_feat_rd_en : 1'b0;

  assign feat_rd_local_addr =
    (le_state == LE_RUN_L1D) ? l1_feat_rd_addr :
    (le_state == LE_RUN_L2P) ? l2_feat_rd_addr :
    (le_state == LE_RUN_AP)  ? ap_feat_rd_addr :
    (le_state == LE_RUN_FC)  ? fc_feat_rd_addr : 16'd0;

  // ============================================================
  // 5) PW channel mux (L0 / L2 / FC)  ---  count λ
  // ============================================================
  wire        l0_pw_req, l2_pw_req, fc_pw_req;
  wire [ADDR_W-1:0] l0_pw_base, l2_pw_base, fc_pw_base;
  wire [16:0]       l0_pw_count, l2_pw_count, fc_pw_count;

  assign pw_req =
    (le_state == LE_RUN_L0)  ? l0_pw_req :
    (le_state == LE_RUN_L2P) ? l2_pw_req :
    (le_state == LE_RUN_FC)  ? fc_pw_req : 1'b0;

  assign pw_base =
    (le_state == LE_RUN_L0)  ? l0_pw_base :
    (le_state == LE_RUN_L2P) ? l2_pw_base :
    (le_state == LE_RUN_FC)  ? fc_pw_base : w_base;

  assign pw_count =
    (le_state == LE_RUN_L0)  ? l0_pw_count :
    (le_state == LE_RUN_L2P) ? l2_pw_count :
    (le_state == LE_RUN_FC)  ? fc_pw_count : 17'd0;

  // ============================================================
  // 6) L0: 3x3 CONV
  // ============================================================
  wire        l0_win_req, l0_win_valid;
  wire [215:0] l0_win_flat;

  window_fetcher_pull_3x3x3_str2_int8 u_l0_fetch (
    .CLK        (CLK),
    .RESET      (RESETn),
    .start_frame(l0_start),
    .win_req    (l0_win_req),
    .win_valid  (l0_win_valid),
    .win_flat   (l0_win_flat),
    .frame_done ()
  );

  conv1_scheduler u_l0_sched (
    .CLK(CLK),
    .RESET(RESETn),
    .start(le_state == LE_RUN_L0),
    .done(),

    .w_base_in(w_base[ADDR_W-1:0]),
    .win_req(l0_win_req),
    .win_valid(l0_win_valid),
    .win_flat(l0_win_flat),

    .weight_req(l0_pw_req),
    .weight_grant(pw_grant),
    .weight_base(l0_pw_base),
    .weight_count(l0_pw_count[10:0]),   
    .weight_valid(pw_valid),
    .weight_data(pw_data),
    .weight_done(pw_done),

    .arr_W_EN(l0_wen),
    .in_weight_above(l0_w32),
    .active_left(l0_act),
    .out_sum_final(arr_out_sum),

    .y_valid(l0_y_valid),
    .y_data(l0_y_data),
    .y_tile_sel()
  );

  // L0 write-back (32 lanes -> two 128b writes)
  reg [15:0] l0_wr_ptr_reg;
  reg        l0_done_reg;
  reg        l0_wb_phase;         // 0: low16, 1: high16
  reg [255:0] l0_q_latched;
  reg        l0_wb_busy;

  wire l0_wb_start = (le_state == LE_RUN_L0) && q_out_valid && !l0_wb_busy && !l0_done_reg;

  assign l0_feat_wr_en   = (le_state == LE_RUN_L0) && l0_wb_busy; // write while busy
  assign l0_feat_wr_addr = l0_wr_ptr_reg + (l0_wb_phase ? 16'd1 : 16'd0);
  assign l0_feat_wr_data = l0_wb_phase ? l0_q_latched[255:128] : l0_q_latched[127:0];
  assign l0_done         = l0_done_reg;

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      l0_wr_ptr_reg <= 16'd0;
      l0_done_reg   <= 1'b0;
      l0_wb_phase   <= 1'b0;
      l0_q_latched  <= 256'd0;
      l0_wb_busy    <= 1'b0;
    end else if (l0_start) begin
      l0_wr_ptr_reg <= 16'd0;
      l0_done_reg   <= 1'b0;
      l0_wb_phase   <= 1'b0;
      l0_q_latched  <= 256'd0;
      l0_wb_busy    <= 1'b0;
    end else begin
      if (l0_wb_start) begin
        l0_q_latched <= q_out_data;
        l0_wb_busy  <= 1'b1;
        l0_wb_phase <= 1'b0;
      end else if (l0_wb_busy) begin
        // we generate one write per cycle through l0_feat_wr_en
        if (l0_wb_phase == 1'b0) begin
          l0_wb_phase <= 1'b1;
        end else begin
          // finish 2nd write and advance pointer by 2 words
          l0_wb_busy  <= 1'b0;
          l0_wb_phase <= 1'b0;
          // keep your old stop-at-1024 behavior, but now step is +2
          if (l0_wr_ptr_reg >= 16'd1022) begin
            l0_done_reg <= 1'b1;
          end else begin
            l0_wr_ptr_reg <= l0_wr_ptr_reg + 16'd2;
          end
        end
      end
    end
  end

  // ============================================================
  // 7) L2: 1x1 PW
  // ============================================================
  wire l2_done_sched;

  pw_scheduler #(
    .NUM_ROWS(PE_ROWS),
    .NUM_COLS(PE_COLS),
    .ADDR_W(ADDR_W),
    .FAST_SIM_EN(FAST_SIM_EN),
    .FAST_COUT_SUBSAMPLE(FAST_PW_COUT_SUBSAMPLE),
    .FAST_PX_SUBSAMPLE(FAST_PW_PX_SUBSAMPLE)
  ) u_l2_sched (
    .CLK(CLK),
    .RESET(RESETn),
    .start(l2_start),
    .done(l2_done_sched),

    .cin(cin),
    .cout(cout),
    .img_w(img_w),
    .img_h(img_h),
    .w_base_in(w_base),

    .weight_req(l2_pw_req),
    .weight_grant(pw_grant),
    .weight_base(l2_pw_base),
    .weight_count(l2_pw_count[10:0]),   // ?? 11b??
    .weight_valid(pw_valid),
    .weight_data(pw_data),
    .weight_done(pw_done),

    .feat_rd_en(l2_feat_rd_en),
    .feat_rd_addr(l2_feat_rd_addr),
    .feat_rd_data(feat_rd_data),
    .feat_rd_valid(feat_rd_valid),

    .arr_W_EN(l2_wen),
    .in_weight_above(l2_w32),
    .active_left(l2_act),
    .out_sum_final(arr_out_sum),

    .y_valid(l2_y_valid),
    .y_data(l2_y_data),
    .y_tile_sel()
  );



  // PW write-back 
  reg  [15:0] l2_wr_ptr_reg;
  reg         l2_done_reg;
  reg         l2_wb_phase;
  reg [255:0] l2_q_latched;
  reg         l2_wb_busy;

  wire        pw_wr_event = (le_state == LE_RUN_L2P) && q_out_valid;
  wire        l2_wb_start = pw_wr_event && !l2_wb_busy && !l2_done_reg;


  localparam integer PW_FALLBACK_TOTAL_WORDS = 50176;
  wire [15:0] pw_last_word_idx =
    (PW_WRITE_LIMIT_EN != 0) ? (PW_WRITE_LIMIT_WORDS[15:0] - 16'd1)
                             : (PW_FALLBACK_TOTAL_WORDS[15:0] - 16'd1);

  assign l2_feat_wr_en   = (le_state == LE_RUN_L2P) && l2_wb_busy;
  assign l2_feat_wr_addr = l2_wr_ptr_reg + (l2_wb_phase ? 16'd1 : 16'd0);
  assign l2_feat_wr_data = l2_wb_phase ? l2_q_latched[255:128] : l2_q_latched[127:0];

  assign l2_done = (FAST_SIM_EN != 0) ? l2_done_sched : l2_done_reg;

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      l2_wr_ptr_reg <= 16'd0;
      l2_done_reg   <= 1'b0;
      l2_wb_phase   <= 1'b0;
      l2_q_latched  <= 256'd0;
      l2_wb_busy    <= 1'b0;      
    end else if (l2_start) begin
      l2_wr_ptr_reg <= 16'd0;
      l2_done_reg   <= 1'b0;
      l2_wb_phase   <= 1'b0;
      l2_q_latched  <= 256'd0;
      l2_wb_busy    <= 1'b0;
    end else begin
      if (l2_wb_start) begin
        l2_q_latched <= q_out_data;
        l2_wb_busy   <= 1'b1;
        l2_wb_phase  <= 1'b0;
      end else if (l2_wb_busy) begin
        if (l2_wb_phase == 1'b0) begin
          l2_wb_phase <= 1'b1;
        end else begin
          l2_wb_busy  <= 1'b0;
          l2_wb_phase <= 1'b0;
          // stop index logic is word-based; now we consume 2 words per event
          if (l2_wr_ptr_reg >= (pw_last_word_idx - 16'd1))
            l2_done_reg <= 1'b1;
          else
            l2_wr_ptr_reg <= l2_wr_ptr_reg + 16'd2;
        end
      end
    end
  end


  // ============================================================
  // 8) L1: DW 3x3
  // ============================================================
  wire l1_cache_load_start, l1_cache_load_done;
  reg  l1_weight_loaded;
    // done rise detect
    

// prefetch & scanner
  wire        l1_buffer_ready, l1_prefetch_done, l1_prefetch_busy;
  wire        l1_read_enable;
  wire [6:0]  l1_read_addr_internal;
  wire [767:0] l1_buffer_out;
  wire        l1_scanner_done, l1_scanner_busy;

  reg  l1_scanner_done_d1;
  wire l1_scanner_done_rise = l1_scanner_done & ~l1_scanner_done_d1;


  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      l1_weight_loaded <= 1'b0;
    end else begin
      // ? 修复：只在scanner完成后才清零weight_loaded，准备下一个block
      if (l1_scanner_done_rise && ! dw_last_block) begin
        l1_weight_loaded <= 1'b0;  // 当前block扫描完成，清零准备下一个
      end else if (l1_cache_load_done) begin
        l1_weight_loaded <= 1'b1;  // 权重加载完成
      end else if (le_state != LE_RUN_L1D) begin
        l1_weight_loaded <= 1'b0;  // 离开DW层时重置
      end
    end
  end

  assign l1_cache_load_start = (le_state == LE_RUN_L1D) && !l1_weight_loaded;

  wire [ADDR_W-1:0] dw_w_base_blk = w_base + (dw_block_idx * 11'd9);

  wire        l1_w_valid;
  wire [3:0]  l1_w_idx;
  wire [127:0] l1_w_data;

  dw_weight_cache #(
    .ADDR_W(ADDR_W)
  ) u_l1_dw_cache (
    .clk           (CLK),
    .rst_n         (RESETn),
    .load_start    (l1_cache_load_start),
    .base_addr     (dw_w_base_blk[ADDR_W-1:0]),
    .load_done     (l1_cache_load_done),

    .ldr_req       (dw_req),
    .ldr_grant     (dw_grant),
    .ldr_base_addr (dw_base),
    .ldr_count     (dw_count),
    .ldr_valid     (dw_valid),
    .ldr_data      (dw_data),
    .ldr_done_sig  (dw_done),

    .w_valid       (l1_w_valid),
    .w_idx         (l1_w_idx),
    .w_data        (l1_w_data)
  );



  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn)
      l1_scanner_done_d1 <= 1'b0;
    else
      l1_scanner_done_d1 <= l1_scanner_done;
  end

  // DW block start condition
  assign dw_block_start =
    (le_state == LE_RUN_L1D) &&
    l1_weight_loaded &&
    bias_valid &&
    ! dw_block_started;

  // block started latch
  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      dw_block_started <= 1'b0;
    end else begin
      if (le_state != LE_RUN_L1D)
        dw_block_started <= 1'b0;
      else if (l1_scanner_done_rise)
        dw_block_started <= 1'b0;
      else if (dw_block_start)
        dw_block_started <= 1'b1;
    end
  end

  

// ============================================================
// ? 修正：输出空间尺寸（这是最终要输出的尺寸）
// ============================================================
wire [7:0] dw_out_w = dw_stride2 ? ((img_w + 8'd1) >> 1) : img_w;
wire [7:0] dw_out_h = dw_stride2 ? ((img_h + 8'd1) >> 1) : img_h;

// ============================================================
// ? Scanner 配置：扫描输入空间（不含padding）
// ============================================================
wire [$clog2(112)-1:0] l1_current_tile_row;
wire l1_tile_start;

wire [7:0] scanner_input_w = img_w + 8'd2;  // 输入宽度 + 左右padding
wire [7:0] scanner_input_h = img_h;          // 输入高度

simple_column_scanner_pipeline #(
  .OUT_W(112),
  .OUT_H(112),
  .TILE_H(6),
  .K(3),
  .PADDING(1)
) u_l1_scanner (
  .clk              (CLK),
  .rst_n            (RESETn),
  .start            (dw_block_start),
  
  // ? 使用原始输入尺寸（scanner内部会处理stride）
  .cfg_w            (scanner_input_w),  // 修复：输入空间宽度
  .cfg_h            (img_h),             // 修复：输入空间高度
  .stride2_en       (dw_stride2),        // ? 修复：传入正确的stride标志
  
  .current_tile_row (l1_current_tile_row),
  .tile_start       (l1_tile_start),
  .buffer_ready     (l1_buffer_ready),
  .read_enable      (l1_read_enable),
  .read_addr        (l1_read_addr_internal),
  .busy             (l1_scanner_busy),
  .done             (l1_scanner_done),
  .current_col      ()
);

// ============================================================
// ? Prefetch buffer 配置
// ============================================================
wire [14:0] l1_rd_addr_raw;
wire [31:0] l1_rd_addr_scaled = (l1_rd_addr_raw * dw_num_blocks) + dw_block_idx;
assign l1_feat_rd_addr = l1_rd_addr_scaled[15:0];

wire l1_prefetch_enable = (le_state == LE_RUN_L1D);

prefetch_double_buffer u_l1_prefetch (
  .clk           (CLK),
  .rst_n         (RESETn),
  .prefetch_start(l1_tile_start),
  .tile_row      (l1_current_tile_row),
  .prefetch_enable(l1_prefetch_enable),
  
  // ? 使用输入空间尺寸
  .cfg_w(img_w),
  .cfg_h(img_h),
  
  .mem_en        (l1_feat_rd_en),
  .mem_addr      (l1_rd_addr_raw),
  .mem_dout      (feat_rd_data),
  .mem_valid     (feat_rd_valid),
  . read_enable   (l1_read_enable),
  .read_addr     (l1_read_addr_internal),
  .buffer_out    (l1_buffer_out),
  .buffer_ready  (l1_buffer_ready),
  .prefetch_busy (l1_prefetch_busy),
  .prefetch_done (l1_prefetch_done)
);

// ============================================================
// ? 数据通路（保持不变）
// ============================================================
wire l1_read_fire = (l1_read_enable && l1_buffer_ready);
reg  l1_read_fire_d1;

always @(posedge CLK or negedge RESETn) begin
  if(!RESETn) l1_read_fire_d1 <= 1'b0;
  else        l1_read_fire_d1 <= l1_read_fire;
end

wire [767:0] l1_column_data;
wire         l1_column_valid;

column_passthrough u_l1_col_pass (
  .clk             (CLK),
  .rst_n           (RESETn),
  .column_data_in  (l1_buffer_out),
  .column_valid    (l1_read_fire_d1),
  .column_data_out (l1_column_data),
  .out_valid       (l1_column_valid)
);


  // DW block index advance ( done_rise ?)
  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      dw_block_idx <= 7'd0;
    end else begin
      if (le_state == LE_IDLE && start && is_dw_layer) begin
        dw_block_idx <= 7'd0;
      end else if (le_state == LE_RUN_L1D && l1_scanner_done_rise) begin
        if (!dw_last_block)
          dw_block_idx <= dw_block_idx + 7'd1;
      end
    end
  end

  // wait weights_ready (?)
  reg [3:0] wstream_cnt;
  reg       weights_ready;

  always @(posedge CLK or negedge RESETn) begin
    if (!RESETn) begin
      wstream_cnt   <= 0;
      weights_ready <= 0;
    end else begin
      if (dw_block_start) begin
        wstream_cnt   <= 0;
        weights_ready <= 0;
      end else if (l1_w_valid && !weights_ready) begin
        if (wstream_cnt == 8)
          weights_ready <= 1;
        else
          wstream_cnt <= wstream_cnt + 1;
      end
    end
  end

  // DW compute
  wire [2047:0] l1_dwc_out_sums;
  wire [63:0]   l1_dwc_out_valids;

  wire any_fifo_full;

  dwc_pu u_l1_dwc (
    .clk        (CLK),
    .rst_n      (RESETn),
    .in_valid   (l1_column_valid && l1_weight_loaded),
    .column_data(l1_column_data),

    .w_valid    (l1_w_valid),
    .w_idx      (l1_w_idx),
    .w_data     (l1_w_data),

    .out_sums   (l1_dwc_out_sums),
    .out_valids (l1_dwc_out_valids)
  );

wire [7:0] qcfg_out_w = dw_stride2 ? ((img_w + 8'd1) >> 1) : img_w;
wire [7:0] qcfg_out_h = dw_stride2 ? ((img_h + 8'd1) >> 1) : img_h;

// ============================================================
// DW 坐标计算和边界检查（修复版本）
// ============================================================

// 1. 延迟3个周期对齐DWC pipeline
reg [6:0] l1_current_col_d1, l1_current_col_d2, l1_current_col_d3;
reg [6:0] l1_tile_row_d1, l1_tile_row_d2, l1_tile_row_d3;

always @(posedge CLK) begin
  // 列号直接来自scanner
  l1_current_col_d1 <= l1_read_addr_internal;
  l1_current_col_d2 <= l1_current_col_d1;
  l1_current_col_d3 <= l1_current_col_d2;
  
  // 行号来自scanner的tile_row
  l1_tile_row_d1 <= l1_current_tile_row;
  l1_tile_row_d2 <= l1_tile_row_d1;
  l1_tile_row_d3 <= l1_tile_row_d2;
end

wire [6:0] l1_current_col = l1_current_col_d3;
wire [6:0] l1_aligned_tile_row = l1_tile_row_d3;

// 2. 计算输入坐标（去除padding）
// Scanner空间：0 到 img_w+1（共img_w+2列）
// 输入空间：-1 到 img_w（-1和img_w是padding）
wire signed [8:0] l1_col_unpadded = {2'b00, l1_current_col} - 9'sd1;
wire signed [8:0] l1_row_unpadded = {2'b00, l1_aligned_tile_row};

// 3. 计算输出坐标（考虑stride）
wire signed [8:0] l1_output_col_signed;
wire signed [8:0] l1_output_row_signed;

assign l1_output_col_signed = dw_stride2 ? (l1_col_unpadded >>> 1) : l1_col_unpadded;
assign l1_output_row_signed = dw_stride2 ? (l1_row_unpadded >>> 1) : l1_row_unpadded;

wire [7:0] l1_output_col = l1_output_col_signed[7:0];
wire [7:0] l1_output_row = l1_output_row_signed[7:0];

// 4. 计算输出feature map的实际尺寸
wire [7:0] expected_out_w = dw_stride2 ? ((img_w + 8'd1) >> 1) : img_w;
wire [7:0] expected_out_h = dw_stride2 ? ((img_h + 8'd1) >> 1) : img_h;

// 5. 边界检查（关键修复）
// 检查1：输出坐标必须非负
wire l1_output_nonneg = (l1_output_col_signed[8] == 1'b0) &&  // 非负数
                        (l1_output_row_signed[8] == 1'b0);

// 检查2：输出坐标必须在输出尺寸范围内
wire l1_output_in_range = (l1_output_col < expected_out_w) &&
                          (l1_output_row < expected_out_h);

// 检查3：stride=2时，只有对齐到grid的输入才产生输出
// （输入的偶数坐标才参与stride=2的卷积）
wire l1_on_stride_grid = ! dw_stride2 || 
                         ((l1_col_unpadded[0] == 1'b0) && 
                          (l1_row_unpadded[0] == 1'b0));

// 检查4：输入坐标的合理性检查（防止scanner异常）
// 允许padding范围：-1 到 img_w（列），0 到 img_h-1（行）
wire l1_input_reasonable = (l1_col_unpadded >= -9'sd1) && 
                           (l1_col_unpadded <= {1'b0, img_w}) &&
                           (l1_row_unpadded >= 9'sd0) &&
                           (l1_row_unpadded < {1'b0, img_h});

// 最终的坐标有效信号（所有条件必须同时满足）
wire l1_coord_valid = l1_output_nonneg &&      // 输出坐标非负
                      l1_output_in_range &&     // 输出坐标在范围内
                      l1_on_stride_grid &&      // 在stride grid上
                      l1_input_reasonable;      // 输入坐标合理

// 6. 过滤DWC输出的valid信号
wire [63:0] l1_filtered_valids = l1_coord_valid ? l1_dwc_out_valids : 64'd0;

// ============================================================
// 传递到Quantization模块
// ============================================================
wire [6:0] q_tile_row  = l1_output_row[6:0];  // 使用输出坐标
wire [6:0] q_col_idx   = l1_output_col[6:0];  // 使用输出坐标
wire [5:0] q_blk_idx   = dw_block_idx[5:0];

quant_l1_stream_4channel #(
  . UNIT_NUM(16),
  .OUT_W_MAX(112),
  .OUT_H_MAX(112),
  .BLOCKS_MAX(64),
  .ACC_W(32),
  .OUT_BITS(8)
) u_dw_quant_stream (
  .clk(CLK),
  .rst_n(RESETn),
  
  // ? 使用输出空间尺寸
  .cfg_out_w(dw_out_w[6:0]),
  .cfg_out_h(dw_out_h[6:0]),
  . cfg_blocks(qcfg_blocks),
  
  . dwc_sums(l1_dwc_out_sums),
  .dwc_valids(l1_filtered_valids),  // ? 使用过滤后的valid信号
  .tile_row(q_tile_row),
  .col_index(q_col_idx),
  .block_idx(q_blk_idx),
  
  .bias_vec(bias_vec),
  .cfg_mult_scalar(quant_M),
  .cfg_shift_scalar(quant_s),
  .cfg_symmetric(1'b0),
  .cfg_zp_out(quant_zp),
  
  .wr_en0(dw_wr_en0), . wr_addr0(dw_wr_addr0), .wr_data0(dw_wr_data0),
  .wr_en1(dw_wr_en1), .wr_addr1(dw_wr_addr1), .wr_data1(dw_wr_data1),
  .wr_en2(dw_wr_en2), .wr_addr2(dw_wr_addr2), .wr_data2(dw_wr_data2),
  .wr_en3(dw_wr_en3), .wr_addr3(dw_wr_addr3), .wr_data3(dw_wr_data3)
);
// ============================================================
  //   // ------------------------------------------------------------
  // DW quant writeback (4-lane) -> Feature buffer
  // ------------------------------------------------------------
  assign l1_feat_wr_en_vec   = {dw_wr_en3,   dw_wr_en2,   dw_wr_en1,   dw_wr_en0};
  assign l1_feat_wr_addr_vec = {dw_wr_addr3, dw_wr_addr2, dw_wr_addr1, dw_wr_addr0};
  assign l1_feat_wr_data_vec = {dw_wr_data3, dw_wr_data2, dw_wr_data1, dw_wr_data0};

assign l1_done = (le_state == LE_RUN_L1D) && l1_scanner_done_rise && dw_last_block;
 // ============================================================
  // 9) AP
  // ============================================================
  global_avg_pool #(
    .CHANNELS  (1024),
    .POOL_SIZE (7),
    .DATA_W    (8),
    .ACC_W     (32),
    .LANES     (16)
  ) u_gap (
    .clk               (CLK),
    .rst_n             (RESETn),
    .start             (ap_start),
    .feat_rd_en        (ap_feat_rd_en),
    .feat_rd_local_addr(ap_feat_rd_addr),
    .feat_rd_data      (feat_rd_data),
    .feat_rd_valid     (feat_rd_valid),
    .feat_wr_en        (ap_feat_wr_en),
    .feat_wr_local_addr(ap_feat_wr_addr),
    .feat_wr_data      (ap_feat_wr_data),
    .done              (ap_done)
  );

  // ============================================================
  // 10) FC
  // ============================================================
  fc_layer #(
    .IN_FEATURES (1024),
    .OUT_CLASSES (1000),
    .DATA_W      (8),
    .ACC_W       (32),
    .LANES       (16),
    .ADDR_W      (ADDR_W)
  ) u_fc (
    .clk               (CLK),
    .rst_n             (RESETn),
    .start             (fc_start),
    .w_base            (w_base),
    .b_base            (b_base),
    .quant_M           (quant_M),
    .quant_s           (quant_s),
    .quant_zp          (quant_zp),

    .weight_req        (fc_pw_req),
    .weight_base       (fc_pw_base),
    .weight_count      (fc_pw_count),
    .weight_grant      (pw_grant),
    .weight_valid      (pw_valid),
    .weight_data       (pw_data),
    .weight_done       (pw_done),

    .bias_vec          (bias_vec),
    .bias_valid        (bias_valid),
    .bias_block_idx    (fc_bias_block_idx),
    .bias_rd_en        (),  // FSM 

    .feat_rd_en        (fc_feat_rd_en),
    .feat_rd_local_addr(fc_feat_rd_addr),
    .feat_rd_data      (feat_rd_data),
    .feat_rd_valid     (feat_rd_valid),

    .out_valid         (fc_out_valid),
    .out_class_idx     (fc_out_class_idx),
    .out_logit         (fc_out_logit),
    .done              (fc_done)
  );

  assign fc_feat_wr_en   = 1'b0;
  assign fc_feat_wr_addr = 16'd0;
  assign fc_feat_wr_data = 128'd0;

`ifndef SYNTHESIS
  // ========== CONV1/PW ?????????? ==========
  integer pe_active_cycles;
  integer pe_total_macs;
  integer pe_valid_outputs;
  
  always @(posedge CLK) begin
    if (!RESETn) begin
      pe_active_cycles <= 0;
      pe_total_macs    <= 0;
      pe_valid_outputs <= 0;
    end else begin
      if (arr_w_en) begin
        pe_active_cycles <= pe_active_cycles + 1;
        pe_total_macs <= pe_total_macs + 1024;
      end
      
      if (q_out_valid) begin
        pe_valid_outputs <= pe_valid_outputs + 32;
      end
    end
  end
  
  // ========== DWC 性能统计（修复Feature I/O统计）==========
  integer dw_total_cycles;
  integer dwc_active_cycles;
  integer dwc_total_outputs;
  integer dwc_unit_usage [0:15];
  integer dwc_column_valid_count;
  integer dwc_weight_loading;
  integer dwc_feat_loading;
  integer dwc_feat_writing;
  
wire dw_layer_active = (le_state == LE_RUN_L1D);
  `ifndef SYNTHESIS
  // ========== DW Layer Active 调试 ==========
  always @(posedge CLK) begin
    if (is_dw_layer) begin
      // 监控dw_layer_active的变化
      if (le_state == LE_IDLE && start) begin
        $display("[L%0d] DW Layer Starting:  dw_layer_active will be set to 1", layer_id);
      end
      
      // 监控状态转换
      if (le_state == LE_RUN_L1D && l1_done) begin
        $display("[L%0d] DW Layer Ending: dw_layer_active will be set to 0", layer_id);
      end
    end
  end
`endif

  integer dw_i;
  integer dw_active_sum;
  real dw_avg_utilization;
  real dw_utilization_per_unit [0:15];
  real dw_total_utilization_pct;
  real dw_weight_loading_pct;
  real dw_feat_io_pct;
  real dw_idle_pct;
  real dw_cycles_per_output;
  
  always @(posedge CLK) begin
    if (!RESETn) begin
      dw_total_cycles <= 0;
      dwc_active_cycles <= 0;
      dwc_total_outputs <= 0;
      dwc_column_valid_count <= 0;
      dwc_weight_loading <= 0;
      dwc_feat_loading <= 0;
      dwc_feat_writing <= 0;
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        dwc_unit_usage[dw_i] <= 0;
      end
    end else if (le_state == LE_RUN_L1D) begin
      // ? 原有的统计保持不变
      dw_total_cycles <= dw_total_cycles + 1;
      
      if (l1_column_valid) begin
        dwc_column_valid_count <= dwc_column_valid_count + 1;
      end
      
      if (u_l1_dwc. in_valid) begin
        dwc_active_cycles <= dwc_active_cycles + 1;
      end
      
      if (l1_w_valid) begin
        dwc_weight_loading <= dwc_weight_loading + 1;
      end
      
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        if (|u_l1_dwc.out_valids[dw_i*4 +: 4]) begin
          dwc_unit_usage[dw_i] <= dwc_unit_usage[dw_i] + 1;
          dwc_total_outputs <= dwc_total_outputs + 4;
        end
      end
    end
    
    // ? 修复：Feature I/O统计扩展到整个DW层活跃期间
    if (dw_layer_active) begin
      if (l1_feat_rd_en) begin
        dwc_feat_loading <= dwc_feat_loading + 1;
      end
      
      if (|l1_feat_wr_en_vec) begin
        dwc_feat_writing <= dwc_feat_writing + 1;
      end
    end
    
    // ? 报告时计算百分比（分母用dw_total_cycles + 预取时间估算）
    if (le_state == LE_RUN_L1D && l1_done) begin
      dw_active_sum = 0;
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        dw_utilization_per_unit[dw_i] = (dw_total_cycles > 0) ? 
                                         (dwc_unit_usage[dw_i] * 100.0 / dw_total_cycles) : 0.0;
        dw_active_sum = dw_active_sum + dwc_unit_usage[dw_i];
      end
      
      dw_avg_utilization = (dw_total_cycles > 0) ? 
                           (dw_active_sum * 100.0 / (dw_total_cycles * 16)) : 0.0;
      
      dw_total_utilization_pct = (dw_total_cycles > 0) ?
                                 (dwc_active_cycles * 100.0 / dw_total_cycles) : 0.0;
      
      dw_weight_loading_pct = (dw_total_cycles > 0) ?
                              (dwc_weight_loading * 100.0 / dw_total_cycles) : 0.0;
      
      // ? 修改：Feature I/O百分比可能>100%（因为在total_cycles之外发生）
      dw_feat_io_pct = (dw_total_cycles > 0) ?
                       ((dwc_feat_loading + dwc_feat_writing) * 100.0 / dw_total_cycles) : 0.0;
      
      dw_idle_pct = 100.0 - dw_total_utilization_pct - dw_weight_loading_pct;
      // 注意：不再从idle中减去feat_io，因为它可能在total_cycles之外
      
      dw_cycles_per_output = (dwc_total_outputs > 0) ?
                             (dw_total_cycles * 1.0 / dwc_total_outputs) : 0.0;
      
      $display("╔════════════════════════════════════════════════════╗");
      $display("║  [DW Layer Performance - Detailed Breakdown]      ║");
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  Total Cycles:      %8d (100.00%%)", dw_total_cycles);
      $display("║  Total Outputs:     %8d", dwc_total_outputs);
      $display("║  Avg Utilization:   %6.2f%%  ★ Overall", dw_avg_utilization);
      $display("╠═════════???══════════════════════════════════════════╣");
      $display("║  Breakdown:");
      $display("║    DWC Active:      %8d (%6.2f%%)  ← Computing", 
               dwc_active_cycles, dw_total_utilization_pct);
      $display("║    Weight Loading:  %8d (%6.2f%%)  ← Loading", 
               dwc_weight_loading, dw_weight_loading_pct);
      $display("║    Feature Read:    %8d (%6.2f%%)  ← Prefetch (may overlap)", 
               dwc_feat_loading, 
               (dw_total_cycles > 0) ? (dwc_feat_loading * 100.0 / dw_total_cycles) : 0.0);
      $display("║    Feature Write:   %8d (%6.2f%%)  ← Writeback", 
               dwc_feat_writing,
               (dw_total_cycles > 0) ? (dwc_feat_writing * 100.0 / dw_total_cycles) : 0.0);
      $display("║    Idle:            %8d (%6.2f%%)  ← Waiting", 
               dw_total_cycles - dwc_active_cycles - dwc_weight_loading,
               dw_idle_pct);
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  Performance Metrics:");
      $display("║    Cycles per Output:    %.2f", dw_cycles_per_output);
      $display("║    Column Valid Count:   %0d", dwc_column_valid_count);
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  DWC Unit Usage (Active Cycles / Utilization):    ║");
      
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        $display("║    Unit[%2d]: %6d cycles (%5.2f%%)", 
                 dw_i, dwc_unit_usage[dw_i], dw_utilization_per_unit[dw_i]);
      end
      
      $display("╚════════════════════════════════════════════════════╝");
      $display("");
      $display("DWC Performance Analysis:");
      
      if (dw_avg_utilization < 20.0) begin
        $display("  [WARNING] DWC utilization very low (%.1f%%)!", dw_avg_utilization);
        if (dw_weight_loading_pct > 30.0) begin
          $display("            → Bottleneck: Weight loading (%.1f%%)", dw_weight_loading_pct);
        end
        if (dw_idle_pct > 50.0) begin
          $display("            → Bottleneck:  FSM overhead or scheduling (%.1f%%)", dw_idle_pct);
        end
      end else if (dw_avg_utilization < 40.0) begin
        $display("  [INFO] DWC utilization moderate (%.1f%%)", dw_avg_utilization);
      end else if (dw_avg_utilization < 60.0) begin
        $display("  [GOOD] DWC utilization good (%.1f%%)!", dw_avg_utilization);
      end else begin
        $display("  [EXCELLENT] DWC utilization excellent (%.1f%%)!", dw_avg_utilization);
      end
      
      // ? 新增：Feature I/O统计报告
      if (dwc_feat_loading > 0) begin
        $display("  [INFO] Feature reads:   %0d cycles (%.1f%% of total, prefetch overlapped)", 
                 dwc_feat_loading, 
                 (dw_total_cycles > 0) ? (dwc_feat_loading * 100.0 / dw_total_cycles) : 0.0);
      end else begin
        $display("  [ERROR] Feature reads: 0 cycles - This is a statistics bug!");
      end
      
      if (dwc_feat_writing > 0) begin
        $display("  [INFO] Feature writes: %0d cycles (%.1f%%)", 
                 dwc_feat_writing,
                 (dw_total_cycles > 0) ? (dwc_feat_writing * 100.0 / dw_total_cycles) : 0.0);
      end
      
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        if (dwc_unit_usage[dw_i] == 0) begin
          $display("  [WARNING] Unit[%2d] not used!   Check channel mapping", dw_i);
        end
      end
      
      $display("");
      
      dw_total_cycles <= 0;
      dwc_active_cycles <= 0;
      dwc_total_outputs <= 0;
      dwc_column_valid_count <= 0;
      dwc_weight_loading <= 0;
      dwc_feat_loading <= 0;
      dwc_feat_writing <= 0;
      for (dw_i=0; dw_i<16; dw_i=dw_i+1) begin
        dwc_unit_usage[dw_i] <= 0;
      end
    end
  end
  
  // ========== CONV1 ???? ==========
  always @(posedge CLK) begin
    if (le_state == LE_RUN_L0 && l0_done) begin
      $display("╔═══════════════════════════════════════════════════╗");
      $display("║  [CONV1 Layer Stats]");
      $display("║  PE Active Cycles:  %0d", pe_active_cycles);
      $display("║  Total MACs:        %0d", pe_total_macs);
      $display("║  Valid Outputs:     %0d", pe_valid_outputs);
      $display("║  Utilization:       %.2f%%", 
               (pe_active_cycles * 100.0) / (pe_active_cycles + 1));
      $display("╚═══════════════════════════════════════════════════╝");
      
      pe_active_cycles <= 0;
      pe_total_macs    <= 0;
      pe_valid_outputs <= 0;
    end
  end
  
  // ========== PW Layer 2 ??????? ==========
  integer l2_y_valid_count;
  integer l2_q_en_count;
  integer l2_q_out_valid_count;
  
  always @(posedge CLK) begin
    if (!RESETn) begin
      l2_y_valid_count <= 0;
      l2_q_en_count <= 0;
      l2_q_out_valid_count <= 0;
    end else if (le_state == LE_RUN_L2P) begin
      if (l2_y_valid) begin
        l2_y_valid_count <= l2_y_valid_count + 1;
      end
      
      if (q_en) begin
        l2_q_en_count <= l2_q_en_count + 1;
      end
      
      if (q_out_valid) begin
        l2_q_out_valid_count <= l2_q_out_valid_count + 1;
      end
    end
    
    if (le_state == LE_RUN_L2P && l2_done) begin
      $display("========================================");
      $display("  [PW Layer 2 Diagnostic]");
      $display("  l2_y_valid count:     %0d", l2_y_valid_count);
      $display("  q_en count:         %0d", l2_q_en_count);
      $display("  q_out_valid count:  %0d", l2_q_out_valid_count);
      $display("========================================");
      
      l2_y_valid_count <= 0;
      l2_q_en_count <= 0;
      l2_q_out_valid_count <= 0;
    end
  end
  
  // ========== PW ??????????? ==========
  integer pw_total_cycles;
  integer pw_weight_wait;
  integer pw_weight_loading;
  integer pw_feat_wait;
  integer pw_feat_loading;
  integer pw_pe_trigger;
  integer pw_pe_pipeline_est;
  integer pw_pe_capture_est;
  integer pw_quant;
  integer pw_writeback;
  
  integer pw_accounted;
  integer pw_unaccounted;
  real pw_weight_wait_pct;
  real pw_weight_loading_pct;
  real pw_feat_io_pct;
  real pw_pe_total_pct;
  real pw_quant_pct;
  real pw_writeback_pct;
  real pw_fsm_overhead_pct;
  
  always @(posedge CLK) begin
    if (!RESETn) begin
      pw_total_cycles <= 0;
      pw_weight_wait <= 0;
      pw_weight_loading <= 0;
      pw_feat_wait <= 0;
      pw_feat_loading <= 0;
      pw_pe_trigger <= 0;
      pw_quant <= 0;
      pw_writeback <= 0;
    end else if (le_state == LE_RUN_L2P) begin
      pw_total_cycles <= pw_total_cycles + 1;
      
      if (l2_pw_req && ! pw_grant) pw_weight_wait <= pw_weight_wait + 1;
      if (pw_valid) pw_weight_loading <= pw_weight_loading + 1;
      if (l2_feat_rd_en && !feat_rd_valid) pw_feat_wait <= pw_feat_wait + 1;
      if (feat_rd_valid) pw_feat_loading <= pw_feat_loading + 1;
      if (arr_w_en) pw_pe_trigger <= pw_pe_trigger + 1;
      if (q_en) pw_quant <= pw_quant + 1;
      if (l2_feat_wr_en) pw_writeback <= pw_writeback + 1;
      
    end else if (le_state != LE_RUN_L2P && pw_total_cycles > 0) begin
      pw_pe_pipeline_est = pw_pe_trigger * 31;
      pw_pe_capture_est  = pw_pe_trigger * 4;
      
      pw_accounted = pw_weight_wait + pw_weight_loading +
                    pw_feat_wait + pw_feat_loading +
                    pw_pe_trigger + pw_pe_pipeline_est + pw_pe_capture_est +
                    pw_quant + pw_writeback;
      
      pw_unaccounted = pw_total_cycles - pw_accounted;
      
      pw_weight_wait_pct    = (pw_total_cycles > 0) ? (pw_weight_wait * 100.0 / pw_total_cycles) : 0.0;
      pw_weight_loading_pct = (pw_total_cycles > 0) ? (pw_weight_loading * 100.0 / pw_total_cycles) : 0.0;
      pw_feat_io_pct        = (pw_total_cycles > 0) ? ((pw_feat_wait + pw_feat_loading) * 100.0 / pw_total_cycles) : 0.0;
      pw_pe_total_pct       = (pw_total_cycles > 0) ? ((pw_pe_trigger + pw_pe_pipeline_est + pw_pe_capture_est) * 100.0 / pw_total_cycles) : 0.0;
      pw_quant_pct          = (pw_total_cycles > 0) ? (pw_quant * 100.0 / pw_total_cycles) : 0.0;
      pw_writeback_pct      = (pw_total_cycles > 0) ? (pw_writeback * 100.0 / pw_total_cycles) : 0.0;
      pw_fsm_overhead_pct   = (pw_total_cycles > 0) ? (pw_unaccounted * 100.0 / pw_total_cycles) : 0.0;
      
      $display("╔════════════════════════════════════════════════════╗");
      $display("║  [PW Layer Performance - CORRECTED Breakdown]     ║");
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  Total Cycles:      %8d (100.00%%)", pw_total_cycles);
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  Weight Wait:       %8d (%6.2f%%)  ← 仲裁等待", 
               pw_weight_wait, pw_weight_wait_pct);
      $display("║  Weight Loading:    %8d (%6.2f%%)  ← 加载权重", 
               pw_weight_loading, pw_weight_loading_pct);
      $display("║  Feature I/O:       %8d (%6.2f%%)  ← Feature读写", 
               pw_feat_wait + pw_feat_loading, pw_feat_io_pct);
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  PE Array Work:                                    ║");
      $display("║    Trigger:          %8d (%6.2f%%)  ← arr_w_en", 
               pw_pe_trigger, pw_pe_trigger * 100.0 / pw_total_cycles);
      $display("║    Pipeline (est):  %8d (%6.2f%%)  ← 31×iter", 
               pw_pe_pipeline_est, pw_pe_pipeline_est * 100.0 / pw_total_cycles);
      $display("║    Capture (est):   %8d (%6.2f%%)  ← 4×iter", 
               pw_pe_capture_est, pw_pe_capture_est * 100.0 / pw_total_cycles);
      $display("║    ──────────────────────────────────────────────");
      $display("║    Subtotal:        %8d (%6.2f%%)  ★ PE真实工作",
               pw_pe_trigger + pw_pe_pipeline_est + pw_pe_capture_est,
               pw_pe_total_pct);
      $display("╠═???══════════════════════════════════════════════════╣");
      $display("║  Quantization:      %8d (%6.2f%%)", pw_quant, pw_quant_pct);
      $display("║  Write Back:        %8d (%6.2f%%)", pw_writeback, pw_writeback_pct);
      $display("╠════════════════════════════════════════════════════╣");
      $display("║  FSM Overhead:      %8d (%6.2f%%)  ← 可优化", 
               pw_unaccounted, pw_fsm_overhead_pct);
      $display("╚════════════════════════════════════════════════════╝");
      
      $display("");
      $display("Performance Summary:");
      $display("  ? Weight Loading Solved:    %.2f%% (was 50.5%%)", pw_weight_loading_pct);
      $display("  ? PE Utilization:          %.2f%%", pw_pe_total_pct);
      $display("  △ FSM Overhead:         %.2f%% (can be optimized)", pw_fsm_overhead_pct);
      $display("  → Cycles per iteration:    %.1f", pw_total_cycles * 1.0 / pw_pe_trigger);
      
      if (pw_weight_loading_pct > 30.0) begin
        $display("[BOTTLENECK] Weight Loading is the primary bottleneck (%.1f%%)!", 
                 pw_weight_loading_pct);
        $display("             → Suggestion:   Implement weight prefetching");
      end
      
      if (pw_fsm_overhead_pct > 40.0) begin
        $display("[WARNING] Large unaccounted time (%.1f%%)!", 
                 pw_fsm_overhead_pct);
        $display("          → Possible FSM overhead or scheduler inefficiency");
      end
      
      pw_total_cycles <= 0;
      pw_weight_wait <= 0;
      pw_weight_loading <= 0;
      pw_feat_wait <= 0;
      pw_feat_loading <= 0;
      pw_pe_trigger <= 0;
      pw_quant <= 0;
      pw_writeback <= 0;
    end
  end
  
  // ========== PW ???? ==========
  always @(posedge CLK) begin
    if (le_state == LE_RUN_L2P && l2_done) begin
      $display("╔═══════════════════════════════════════════════════╗");
      $display("║  [PW Layer Stats]");
      $display("║  PE Active Cycles:  %0d", pe_active_cycles);
      $display("║  Total MACs:        %0d", pe_total_macs);
      $display("║  Valid Outputs:     %0d", pe_valid_outputs);
      $display("╚═══════════════════════???═══════════════════════════╝");
      
      pe_active_cycles <= 0;
      pe_total_macs    <= 0;
      pe_valid_outputs <= 0;
    end
  end

`endif

endmodule