`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:38:35
// Design Name: 
// Module Name: prefetch_double_buffer
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
module prefetch_double_buffer #(
  parameter integer OUT_W  = 112,
  parameter integer OUT_H  = 112,
  parameter integer TILE_H = 6,
  parameter integer DATA_W = 8,
  parameter integer LANES  = 16
)(
  input  wire clk,
  input  wire rst_n,
  input  wire prefetch_start,
  input  wire [$clog2(OUT_H)-1:0] tile_row,

  // Memory interface
  output reg  mem_en,
  output reg  [$clog2(OUT_W*OUT_H*(32/LANES))-1:0] mem_addr,
  input  wire [LANES*DATA_W-1:0] mem_dout,
  input  wire mem_valid,
  
  input wire [7:0] cfg_w,
  input wire [7:0] cfg_h,
  input  wire prefetch_enable,

  // Consumer interface
  input  wire read_enable,
  input  wire [$clog2(OUT_W)-1:0] read_addr,
  output reg  [TILE_H*DATA_W*LANES-1:0] buffer_out,
  output reg  buffer_ready,
  output wire prefetch_busy,
  output wire prefetch_done
);

  localparam integer WORD_W   = DATA_W * LANES;
  localparam integer PADDING  = 1;
  localparam integer PADDED_W = OUT_W + 2 * PADDING;
  localparam integer ADDR_W   = $clog2(OUT_W*OUT_H*(32/LANES));
  localparam integer BUF_AW   = $clog2(PADDED_W);
  localparam integer ROW_W    = $clog2(TILE_H);
  localparam integer MAX_OUTSTANDING = 64; // ? 降低以减少复杂度
  
  wire [BUF_AW-1:0] padded_w_dyn  = cfg_w + 8'd2;
  wire [15:0]       total_words_dyn = TILE_H * padded_w_dyn;

  // ============================================================
  // ? 核心状态精简
  // ============================================================
  reg active_read_buf;   // 0=A, 1=B
  reg active_write_buf;  // 0=A, 1=B
  reg bufA_valid, bufB_valid;
  
  reg write_active;
  reg prefetch_done_reg;
  
  reg [BUF_AW-1:0] req_colp;
  reg [ROW_W-1:0]  req_rowi;
  reg [$clog2(OUT_H)-1:0] base_tile_row;
  reg [15:0] issue_count;
  reg [15:0] commit_count;
  reg [7:0]  rd_outstanding;

  assign prefetch_busy = write_active;
  assign prefetch_done = prefetch_done_reg;

  // ============================================================
  // ? 简化的 FIFO (单通道，去除4-lane复杂度)
  // ============================================================
  localparam integer TAG_W = 1 + 1 + ROW_W + BUF_AW;
  localparam integer FIFO_DEPTH = 512; // ? 减小深度
  
  reg  [TAG_W-1:0] tag_fifo [0:FIFO_DEPTH-1];
  reg  [9:0] tag_wr_ptr, tag_rd_ptr;
  wire tag_empty = (tag_wr_ptr == tag_rd_ptr);
  wire tag_full  = ((tag_wr_ptr + 1'b1) == tag_rd_ptr);
  
  reg  [WORD_W-1:0] data_fifo [0:FIFO_DEPTH-1];
  reg  [9:0] data_wr_ptr, data_rd_ptr;
  wire data_empty = (data_wr_ptr == data_rd_ptr);
  wire data_full  = ((data_wr_ptr + 1'b1) == data_rd_ptr);

  // Address calculation
  function [ADDR_W-1:0] calc_addr;
    input [$clog2(OUT_H)-1:0] row;
    input [BUF_AW-1:0]        col;
    reg [31:0] temp;
    begin
      temp = row * cfg_w + col;
      calc_addr = temp[ADDR_W-1:0];
    end
  endfunction

  wire [$clog2(OUT_H)-1:0] abs_row = base_tile_row + req_rowi;
  wire is_pad_col = (req_colp == {BUF_AW{1'b0}}) || (req_colp == (padded_w_dyn - 1'b1));
  wire is_row_oob = (abs_row >= cfg_h[$clog2(OUT_H)-1:0]);
  wire is_zero_req = is_pad_col || is_row_oob;
  wire [BUF_AW-1:0] col_ext = req_colp - 1'b1;

  // ============================================================
  // ? Issue & Commit 逻辑
  // ============================================================
  wire can_issue = write_active && (issue_count < total_words_dyn) && ! tag_full;
  wire outstanding_ok = (rd_outstanding < MAX_OUTSTANDING);
  wire mem_issue = can_issue && !is_zero_req && outstanding_ok && ! data_full;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      write_active      <= 1'b0;
      prefetch_done_reg <= 1'b0;
      active_read_buf   <= 1'b0;
      active_write_buf  <= 1'b0;
      bufA_valid        <= 1'b0;
      bufB_valid        <= 1'b0;
      base_tile_row     <= 0;
      req_colp          <= 0;
      req_rowi          <= 0;
      issue_count       <= 0;
      commit_count      <= 0;
      rd_outstanding    <= 0;
      mem_en            <= 1'b0;
      mem_addr          <= 0;
      tag_wr_ptr        <= 0;
      tag_rd_ptr        <= 0;
      data_wr_ptr       <= 0;
      data_rd_ptr       <= 0;
    end else begin
      mem_en            <= 1'b0;
      prefetch_done_reg <= 1'b0;

      // ? Outstanding 管理
      case ({mem_issue, mem_valid && prefetch_enable})
        2'b10: rd_outstanding <= rd_outstanding + 1'b1;
        2'b01: rd_outstanding <= (rd_outstanding > 0) ? (rd_outstanding - 1'b1) : 8'd0;
      endcase

      // ? 启动预取
      if (prefetch_start && !write_active) begin
        write_active   <= 1'b1;
        base_tile_row  <= tile_row;
        req_colp       <= 0;
        req_rowi       <= 0;
        issue_count    <= 0;
        commit_count   <= 0;
        rd_outstanding <= 0;
        
        // ? 选择空闲buffer
        if (! bufA_valid) begin
          active_write_buf <= 1'b0;
          bufA_valid       <= 1'b0;
        end else if (!bufB_valid) begin
          active_write_buf <= 1'b1;
          bufB_valid       <= 1'b0;
        end else begin
          active_write_buf <= ~active_read_buf; // 选择非读buffer
        end
      end

      // ? Issue 逻辑
      if (can_issue) begin
        if (is_zero_req) begin
          // 写入 padding 标记
          tag_fifo[tag_wr_ptr] <= {active_write_buf, 1'b1, req_rowi, req_colp};
          tag_wr_ptr <= tag_wr_ptr + 1'b1;
          issue_count <= issue_count + 1;
        end else if (mem_issue) begin
          // 发起内存读取
          tag_fifo[tag_wr_ptr] <= {active_write_buf, 1'b0, req_rowi, req_colp};
          tag_wr_ptr <= tag_wr_ptr + 1'b1;
          mem_en   <= 1'b1;
          mem_addr <= calc_addr(abs_row, col_ext);
          issue_count <= issue_count + 1;
        end
        
        // 更新行列索引
        if (issue_count + 1 < total_words_dyn) begin
          if (req_rowi < TILE_H-1) 
            req_rowi <= req_rowi + 1;
          else begin 
            req_rowi <= 0; 
            req_colp <= req_colp + 1; 
          end
        end
      end

      // ? Data FIFO 写入
      if (mem_valid && ! data_full && prefetch_enable) begin
        data_fifo[data_wr_ptr] <= mem_dout;
        data_wr_ptr <= data_wr_ptr + 1'b1;
      end

      // ? Commit 完成
      if (write_active && (commit_count == total_words_dyn)) begin
        write_active      <= 1'b0;
        prefetch_done_reg <= 1'b1;
        
        if (active_write_buf == 1'b0)
          bufA_valid <= 1'b1;
        else
          bufB_valid <= 1'b1;
          
        active_write_buf <= ~active_write_buf;
      end
    end
  end

  // ============================================================
  // ? Writer - 简化为单通道
  // ============================================================
  wire [TAG_W-1:0] tag_cur = tag_fifo[tag_rd_ptr];
  wire tag_is_zero = tag_cur[TAG_W-2];
  wire tag_buf_sel = tag_cur[TAG_W-1];
  wire [ROW_W-1:0] tag_rowi = tag_cur[TAG_W-3 -: ROW_W];
  wire [BUF_AW-1:0] tag_colp = tag_cur[BUF_AW-1:0];
  
  wire can_commit = write_active && ! tag_empty && (tag_is_zero || ! data_empty);
  
  reg wr_fire;
  reg wr_buf_sel;
  reg [ROW_W-1:0] wr_rowi;
  reg [BUF_AW-1:0] wr_colp;
  reg [WORD_W-1:0] wr_data;
  
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_fire   <= 1'b0;
      wr_buf_sel <= 1'b0;
      wr_rowi   <= 0;
      wr_colp   <= 0;
      wr_data   <= 0;
      commit_count <= 0;
    end else begin
      wr_fire <= 1'b0;
      
      if (can_commit) begin
        wr_fire    <= 1'b1;
        wr_buf_sel <= tag_buf_sel;
        wr_rowi    <= tag_rowi;
        wr_colp    <= tag_colp;
        wr_data    <= tag_is_zero ? {WORD_W{1'b0}} : data_fifo[data_rd_ptr];
        
        tag_rd_ptr <= tag_rd_ptr + 1'b1;
        if (! tag_is_zero)
          data_rd_ptr <= data_rd_ptr + 1'b1;
        
        commit_count <= commit_count + 1;
      end
      
      if (! write_active)
        commit_count <= 0;
    end
  end

  // ============================================================
  // ? RAM - Dual Port (6 rows × 2 buffers)
  // ============================================================
  wire ram_en_read = (read_enable && buffer_ready && (read_addr[BUF_AW-1:0] < padded_w_dyn));
  
  wire [WORD_W-1:0] rowA_dout [0:TILE_H-1];
  wire [WORD_W-1:0] rowB_dout [0:TILE_H-1];
  
  genvar r;
  generate
    for (r = 0; r < TILE_H; r = r + 1) begin : GEN_ROWS
      wire writeA = wr_fire && (wr_rowi == r) && (wr_buf_sel == 1'b0);
      wire writeB = wr_fire && (wr_rowi == r) && (wr_buf_sel == 1'b1);
      wire readA  = ram_en_read && (active_read_buf == 1'b0);
      wire readB  = ram_en_read && (active_read_buf == 1'b1);
      
      // Buffer A
      xpm_memory_tdpram #(
        .ADDR_WIDTH_A(BUF_AW),
        . ADDR_WIDTH_B(BUF_AW),
        .MEMORY_SIZE(WORD_W * PADDED_W),
        .MEMORY_PRIMITIVE("block"),
        .WRITE_DATA_WIDTH_A(WORD_W),
        .READ_DATA_WIDTH_B(WORD_W),
        .READ_LATENCY_B(1),
        .WRITE_MODE_A("no_change"),
        .WRITE_MODE_B("no_change")
      ) u_bufA_row (
        .clka(clk), .ena(writeA), .wea(1'b1),
        .addra(wr_colp), .dina(wr_data), .douta(),
        .clkb(clk), .enb(readA), .web(1'b0),
        .addrb(read_addr[BUF_AW-1:0]), .dinb({WORD_W{1'b0}}), .doutb(rowA_dout[r]),
        .sleep(1'b0), .rsta(1'b0), .rstb(1'b0), .regcea(1'b1), .regceb(1'b1),
        .injectdbiterra(1'b0), .injectsbiterra(1'b0),
        .injectdbiterrb(1'b0), .injectsbiterrb(1'b0),
        .dbiterrb(), .sbiterrb()
      );
      
      // Buffer B
      xpm_memory_tdpram #(
        . ADDR_WIDTH_A(BUF_AW),
        .ADDR_WIDTH_B(BUF_AW),
        .MEMORY_SIZE(WORD_W * PADDED_W),
        .MEMORY_PRIMITIVE("block"),
        .WRITE_DATA_WIDTH_A(WORD_W),
        .READ_DATA_WIDTH_B(WORD_W),
        .READ_LATENCY_B(1),
        .WRITE_MODE_A("no_change"),
        .WRITE_MODE_B("no_change")
      ) u_bufB_row (
        .clka(clk), .ena(writeB), .wea(1'b1),
        .addra(wr_colp), .dina(wr_data), .douta(),
        .clkb(clk), .enb(readB), .web(1'b0),
        .addrb(read_addr[BUF_AW-1:0]), .dinb({WORD_W{1'b0}}), .doutb(rowB_dout[r]),
        .sleep(1'b0), .rsta(1'b0), .rstb(1'b0), .regcea(1'b1), .regceb(1'b1),
        .injectdbiterra(1'b0), .injectsbiterra(1'b0),
        .injectdbiterrb(1'b0), .injectsbiterrb(1'b0),
        .dbiterrb(), .sbiterrb()
      );
    end
  endgenerate

  // ============================================================
  // ? Buffer ready & output
  // ============================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      buffer_ready <= 1'b0;
      active_read_buf <= 1'b0;
    end else begin
      buffer_ready <= (active_read_buf == 1'b0) ?  bufA_valid : bufB_valid;
      
      // ? 自动切换读buffer（当预取完成时）
      if (prefetch_done_reg) begin
        if (bufA_valid && !bufB_valid)
          active_read_buf <= 1'b0;
        else if (bufB_valid && !bufA_valid)
          active_read_buf <= 1'b1;
        else if (bufA_valid && bufB_valid)
          active_read_buf <= active_write_buf; // 切换到刚写完的buffer
      end
    end
  end
  
  // ? Output mux (延迟1周期对齐RAM读延迟)
  reg read_en_d1;
  reg active_read_buf_d1;
  
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      buffer_out <= {TILE_H*WORD_W{1'b0}};
      read_en_d1 <= 1'b0;
      active_read_buf_d1 <= 1'b0;
    end else begin
      read_en_d1 <= ram_en_read;
      active_read_buf_d1 <= active_read_buf;
      
      if (read_en_d1) begin
        if (active_read_buf_d1)
          buffer_out <= {rowB_dout[5], rowB_dout[4], rowB_dout[3], 
                         rowB_dout[2], rowB_dout[1], rowB_dout[0]};
        else
          buffer_out <= {rowA_dout[5], rowA_dout[4], rowA_dout[3], 
                         rowA_dout[2], rowA_dout[1], rowA_dout[0]};
      end
    end
  end

endmodule