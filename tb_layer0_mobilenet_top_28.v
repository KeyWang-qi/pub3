`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:44:06
// Design Name: 
// Module Name: tb_layer0_mobilenet_top_28
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

module tb_layer0_mobilenet_top_28;
  reg  CLK;
  reg  RESETn;
  reg  start;
  wire done;
  wire [2:0] fsm_state;
  wire [5:0] current_layer;

  //============================================================
  // Clock generation
  //============================================================
  initial begin
    CLK = 1'b0;
    forever #5 CLK = ~CLK;   // 100 MHz
  end

  //============================================================
  // DUT
  //============================================================
  mobilenet_top_28layers#(
    .START_LAYER_ID(6'd0),
    .MAX_LAYER_ID  (6'd28)
  ) dut (
    .CLK          (CLK),
    .RESETn       (RESETn),
    .start        (start),
    .done         (done),
    .fsm_state    (fsm_state),
    .current_layer(current_layer)
  );

  //============================================================
  // Tap internal DUT signals (hierarchical references)
  //============================================================
  wire [2:0] layer_type;
  wire       layer_start_i;
  wire       layer_done_i;

  assign layer_type   = dut.layer_type_cur; // from layer_config_rom
  assign layer_start_i = dut.layer_start;
  assign layer_done_i  = dut.layer_done;
  //============================================================
  // Layer type -> string
  //============================================================
  function [63:0] layer_type_str;
    input [2:0] t;
    begin
      case (t)
        3'd0: layer_type_str = "CONV    ";
        3'd1: layer_type_str = "DW      ";
        3'd2: layer_type_str = "PW      ";
        3'd3: layer_type_str = "AP      ";
        3'd4: layer_type_str = "FC      ";
        default: layer_type_str = "UNKNOWN ";
      endcase
    end
  endfunction

  //============================================================
  // Timeout guard
  //============================================================
  localparam TIMEOUT_NS = 100_000_000;  // 100 ms timeout


  //============================================================
  // Finish flag (done or timeout)
  //============================================================
  reg finish_flag;

  // done 到来，finish_flag=1
  always @(posedge CLK) begin
    if (!RESETn) begin
      finish_flag <= 1'b0;
    end else if (done) begin
      finish_flag <= 1'b1;
    end
  end

  // timeout 到来，finish_flag=1
  initial begin
    finish_flag = 1'b0;
    #TIMEOUT_NS;
    if (finish_flag == 1'b0) begin
      $display("[%0t] [TB] TIMEOUT reached!", $time);
      finish_flag = 1'b1;
    end
  end

  //============================================================
  // Monitor: transitions + per-layer runtime
  //============================================================
  integer cycle_cnt;
  reg [5:0] last_layer;
  reg [2:0] last_fsm;

  reg [5:0] active_layer;
  reg [2:0] active_type;
  time      t_layer_start;
  integer   c_layer_start;

  initial begin
    cycle_cnt     = 0;
    last_layer    = 6'h3F;
    last_fsm      = 3'h7;
    active_layer  = 0;
    active_type   = 0;
    t_layer_start = 0;
    c_layer_start = 0;
  end

  always @(posedge CLK) begin
    if (!RESETn) begin
      cycle_cnt     <= 0;
      last_layer    <= 6'h3F;
      last_fsm      <= 3'h7;
      active_layer  <= 0;
      active_type   <= 0;
      t_layer_start <= 0;
      c_layer_start <= 0;
    end else begin
      cycle_cnt <= cycle_cnt + 1;

      if (layer_start_i) begin
        active_layer  <= current_layer;
        active_type   <= layer_type;
        t_layer_start <= $time;
        c_layer_start <= cycle_cnt;

        $display("[%0t ns][C%0d] >>> L%0d %s START",
                 $time, cycle_cnt, current_layer, layer_type_str(layer_type));
      end

      // 层结束：打印耗时（用 active_layer/active_type 防止同周期 layer 已经变了）
      if (layer_done_i) begin
        $display("[%0t ns][C%0d] <<< L%0d %s DONE   dt=%0t ps   dcy=%0d",
                 $time, cycle_cnt, active_layer, layer_type_str(active_type),
                 ($time - t_layer_start),
                 (cycle_cnt - c_layer_start));
      end

      // 网络结束
      if (done) begin
        $display("[%0t ns][C%0d] NETWORK DONE", $time, cycle_cnt);
      end
    end
  end

  //============================================================
  // Main stimulus
  //============================================================
  initial begin
    $display("========================================");
    RESETn = 1'b0;
    start  = 1'b0;
    #100;

    RESETn = 1'b1;
    $display("[%0t] Reset released", $time);
    #50;

    $display("[%0t] Starting network execution...", $time);
    start = 1'b1;
    #10;
    start = 1'b0;

    // 等待 done 或 timeout
    wait(finish_flag == 1'b1);

    // ? 这里 #2000 绝对不会报错（在 initial 里）
    #2000;

    $display("========================================");
    $display("Simulation Summary:");
    $display("  Final FSM state: %0d", fsm_state);
    $display("  Final layer: %0d", current_layer);
    $display("========================================");
    $finish;
    

  end


endmodule
