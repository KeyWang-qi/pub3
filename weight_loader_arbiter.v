`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:25:01
// Design Name: 
// Module Name: weight_loader_arbiter
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
module weight_loader_arbiter #(
    parameter ADDR_W = 32,
    parameter integer BUF_ADDR_W = 15 
)(
    input  wire         clk,
    input  wire         rst_n,

    // ===== DW Cache  =====
    input  wire         dw_req,
    input  wire [ADDR_W-1:0] dw_base,
    input  wire [16:0]  dw_count,
    output reg          dw_grant,
    output reg          dw_valid,
    output reg  [127:0] dw_data,
    output reg          dw_done,

    // ===== CONV/PW Scheduler =====
    input  wire         pw_req,
    input  wire [ADDR_W-1:0] pw_base,
    input  wire [16:0]  pw_count,
    output reg          pw_grant,
    output reg          pw_valid,
    output reg  [127:0] pw_data,
    output reg          pw_done,

    // ===== preload handshake 透出到 top 连接 DMA =====
    output wire         preload_req,
    output wire [ADDR_W-1:0] preload_base,
    output wire [16:0]  preload_count,
    input  wire         preload_done,

    // ===== 连接到 weight buffer =====
    output wire         bmg_en,
    output wire [BUF_ADDR_W-1:0] bmg_addr, // 自动变为 [14:0]
    input  wire [127:0] bmg_data
);

    // -----------------------------
    // loader control signals
    // -----------------------------
    reg                  ldr_start;
    reg  [ADDR_W-1:0]    ldr_base;
    reg  [18:0]          ldr_count;

    // -----------------------------
    // loader stream signals
    // -----------------------------
    wire                 ldr_valid_i;
    wire [127:0]         ldr_data_i;
    wire                 ldr_done_i;

    // [修改] 实例化 Loader 时传入 BUF_ADDR_W
    weight_loader_universal #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (128),
        .RD_LAT     (2),
        .BUF_ADDR_W (BUF_ADDR_W) // 传入 15
    ) u_loader (
        .clk          (clk),
        .rst_n        (rst_n),

        .start        (ldr_start),
        .base_addr    (ldr_base),
        // 截断高位匹配 loader 接口
        .load_count   (ldr_count[16:0]), 
        .done         (ldr_done_i),

        .preload_req  (preload_req),
        .preload_base (preload_base),
        .preload_count(preload_count[16:0]), // 截断匹配
        .preload_done (preload_done),

        .bmg_en       (bmg_en),
        .bmg_addr     (bmg_addr),
        .bmg_data     (bmg_data),

        .out_valid    (ldr_valid_i),
        .out_data     (ldr_data_i)
    );

    // -----------------------------
    // Arbiter FSM: IDLE / LOADING
    // -----------------------------
    localparam S_IDLE    = 1'b0;
    localparam S_LOADING = 1'b1;

    reg state;
    reg current_channel;   // 0 = DW, 1 = PW
    reg ldr_started;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            current_channel <= 1'b0;

            dw_grant <= 1'b0;
            pw_grant <= 1'b0;

            ldr_start   <= 1'b0;
            ldr_base    <= {ADDR_W{1'b0}};
            ldr_count   <= 19'd0;
            ldr_started <= 1'b0;

            dw_valid <= 1'b0;  dw_data <= 128'd0;  dw_done <= 1'b0;
            pw_valid <= 1'b0;  pw_data <= 128'd0;  pw_done <= 1'b0;

        end else begin

            dw_valid <= 1'b0;
            pw_valid <= 1'b0;
            dw_done  <= 1'b0;
            pw_done  <= 1'b0;

            if (ldr_valid_i || ldr_done_i || preload_req || bmg_en)
                ldr_started <= 1'b1;

            case (state)
                S_IDLE: begin
                    dw_grant    <= 1'b0;
                    pw_grant    <= 1'b0;
                    ldr_start   <= 1'b0;
                    ldr_started <= 1'b0;

                    if (dw_req) begin
                        current_channel <= 1'b0;   // DW
                        dw_grant        <= 1'b1;
                        ldr_base        <= dw_base;
                        ldr_count       <= dw_count;
                        ldr_start       <= 1'b1;
                        state           <= S_LOADING;
                    end else if (pw_req) begin
                        current_channel <= 1'b1;   // PW
                        pw_grant        <= 1'b1;
                        ldr_base        <= pw_base;
                        ldr_count       <= pw_count;
                        ldr_start       <= 1'b1;
                        state           <= S_LOADING;
                    end
                end

                S_LOADING: begin
                    if (!ldr_started)
                        ldr_start <= 1'b1;
                    else
                        ldr_start <= 1'b0;

                    if (ldr_valid_i) begin
                        if (current_channel == 1'b0) begin
                            dw_valid <= 1'b1;
                            dw_data  <= ldr_data_i;
                        end else begin
                            pw_valid <= 1'b1;
                            pw_data  <= ldr_data_i;
                        end
                    end

                    // 完成：根据通道结束并回到 IDLE
                    if (ldr_done_i) begin
                        if (current_channel == 1'b0) begin
                            dw_done  <= 1'b1;
                            dw_grant <= 1'b0;
                        end else begin
                            pw_done  <= 1'b1;
                            pw_grant <= 1'b0;
                        end

                        ldr_start <= 1'b0;
                        state     <= S_IDLE;
                    end
                end

                default: begin
                    state       <= S_IDLE;
                    ldr_start   <= 1'b0;
                    ldr_started <= 1'b0;
                end
            endcase
        end
    end
endmodule