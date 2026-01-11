`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:37:49
// Design Name: 
// Module Name: simple_column_scanner_pipeline
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
module simple_column_scanner_pipeline #(
  parameter integer OUT_W   = 112,
  parameter integer OUT_H   = 112,
  parameter integer TILE_H  = 6,
  parameter integer K       = 3,
  parameter integer PADDING = 1  // ? 改回 1（与双缓冲一致）
)(
  input  wire clk,
  input  wire rst_n,
  input  wire start,

  // Runtime configuration
  input  wire [7:0] cfg_w,
  input  wire [7:0] cfg_h,
  input  wire       stride2_en,

  // ? 恢复这些端口（但含义改变）
  output reg  [$clog2(OUT_H)-1:0] current_tile_row,  // 当前 tile 行
  output reg  tile_start,  // 新 tile 开始脉冲
  input  wire buffer_ready,

  // Read interface
  output wire read_enable,
  output wire [$clog2(OUT_W)-1:0] read_addr,

  // Status
  output reg  busy,
  output reg  done,
  output reg  [$clog2(OUT_W)-1:0] current_col
);

  // ============================================================
  // 常量定义
  // ============================================================
  localparam integer STEP_ROW_S1_INT = (TILE_H - K + 1);      // 4
  localparam integer STEP_ROW_S2_INT = (TILE_H - K + 1) * 2;  // 8
  
  wire [$clog2(OUT_H)-1:0] step_row_s1 = STEP_ROW_S1_INT[$clog2(OUT_H)-1:0];
  wire [$clog2(OUT_H)-1:0] step_row_s2 = STEP_ROW_S2_INT[$clog2(OUT_H)-1:0];

  // ============================================================
  // 状态机（保持 2 个状态）
  // ============================================================
  localparam S_IDLE = 1'b0;
  localparam S_SCAN = 1'b1;

  reg state;
  reg [$clog2(OUT_W)-1:0] col_counter;

  // ============================================================
  // Runtime configuration
  // ============================================================
  wire [7:0] cfg_w_clamped = (cfg_w > OUT_W[7:0]) ? OUT_W[7:0] : cfg_w;
  wire [7:0] cfg_h_clamped = (cfg_h > OUT_H[7:0]) ? OUT_H[7:0] : cfg_h;

  // Padded width
  wire [$clog2(OUT_W)-1:0] padded_w = cfg_w_clamped[$clog2(OUT_W)-1:0] + (PADDING * 2);
  wire [$clog2(OUT_W)-1:0] w_minus1 = padded_w - 1'b1;
  wire [$clog2(OUT_W)-1:0] w_minus2 = (padded_w >= 2) ? (padded_w - 2'd2) : {($clog2(OUT_W)){1'b0}};
  
  wire [$clog2(OUT_H)-1:0] h_limit = cfg_h_clamped[$clog2(OUT_H)-1:0];

// ? 提前预取触发点（80% 位置，预留 20% 重叠时间）
wire [$clog2(OUT_W)-1:0] prefetch_trigger_col;
assign prefetch_trigger_col = (padded_w * 4) / 5;  // 80% 位置

// ? 检测是否应该提前预取下一个 tile
wire should_trigger_prefetch_s1 = (col_counter == prefetch_trigger_col) && 
                                   ((current_tile_row + step_row_s1) < h_limit) && 
                                   (! stride2_en);

wire should_trigger_prefetch_s2 = (col_counter == prefetch_trigger_col) && 
                                   ((current_tile_row + step_row_s2) < h_limit) && 
                                   (stride2_en);

  // ============================================================
  // Output assignments
  // ============================================================
  assign read_enable = (state == S_SCAN) && buffer_ready;
  assign read_addr   = col_counter;

  // ============================================================
  // ? 关键修正：FSM 管理 tile 迭代
  // ============================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state            <= S_IDLE;
      busy             <= 1'b0;
      done             <= 1'b0;
      col_counter      <= {($clog2(OUT_W)){1'b0}};
      current_col      <= {($clog2(OUT_W)){1'b0}};
      current_tile_row <= {($clog2(OUT_H)){1'b0}};
      tile_start       <= 1'b0;
    end else begin
      done       <= 1'b0;  // Pulse
      tile_start <= 1'b0;  // Pulse

      case (state)
        
        // ========================================
        // IDLE: Wait for start
        // ========================================
        S_IDLE: begin
          busy             <= 1'b0;
          col_counter      <= 0;
          current_col      <= 0;
          current_tile_row <= 0;

          if (start) begin
            busy       <= 1'b1;
            tile_start <= 1'b1;  // ? 触发第一个 tile 预取
            state      <= S_SCAN;
          end
        end

        // ========================================
        // SCAN: Scan columns with tile iteration
        // ========================================
// ========== 替换第 106-156 行的 S_SCAN 状态： ==========

S_SCAN: begin
  if (buffer_ready) begin
    current_col <= col_counter;

    // ? 提前预取逻辑：在扫描到 80% 时就触发下一个 tile 的预取
    if (should_trigger_prefetch_s1 || should_trigger_prefetch_s2) begin
      tile_start <= 1'b1;  // ? 提前 20% 就开始预取
    end

    // Stride = 1
    if (! stride2_en) begin
      if (col_counter < w_minus1) begin
        col_counter <= col_counter + 1'b1;
      end else begin
        // 一行扫描完成，更新 tile_row
        col_counter <= 0;
        
        if ((current_tile_row + step_row_s1) < h_limit) begin
          current_tile_row <= current_tile_row + step_row_s1;
          // ? 注意：不再在这里触发 tile_start（已在 80% 时提前触发）
        end else begin
          // 所有 tile 完成
          busy  <= 1'b0;
          done  <= 1'b1;
          state <= S_IDLE;
        end
      end
    end
    
    // Stride = 2
    else begin
      if (col_counter < w_minus2) begin
        col_counter <= col_counter + 2'd2;
      end else begin
        col_counter <= 0;
        
        if ((current_tile_row + step_row_s2) < h_limit) begin
          current_tile_row <= current_tile_row + step_row_s2;
          // ? 不再在这里触发 tile_start
        end else begin
          busy  <= 1'b0;
          done  <= 1'b1;
          state <= S_IDLE;
        end
      end
    end
  end
end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule