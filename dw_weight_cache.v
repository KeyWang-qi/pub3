`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:37:20
// Design Name: 
// Module Name: dw_weight_cache
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
module dw_weight_cache #(
parameter integer ADDR_W = 16,
    parameter UNIT_NUM = 16,
    parameter DATA_W   = 8,
    parameter K        = 3
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        load_start,
    input  wire [18:0] base_addr,
    output reg         load_done,

    output reg         ldr_req,
    input  wire        ldr_grant,
    output wire [18:0] ldr_base_addr,
    output wire [10:0] ldr_count,
    input  wire        ldr_valid,
    input  wire [127:0] ldr_data,
    input  wire        ldr_done_sig,

    // ===== 新输出：tap stream =====
    output reg         w_valid,
    output reg  [3:0]  w_idx,
    output reg  [127:0] w_data
);

    localparam integer KK = K*K; // 9
    assign ldr_base_addr = base_addr;
    assign ldr_count     = 11'd9;

    reg [127:0] tap_buf [0:8];
    reg [3:0]   recv_cnt;
    reg         loading;

    // 输出tap的计数
    reg [3:0]   out_cnt;
    reg         streaming;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            recv_cnt   <= 0;
            loading    <= 0;
            load_done  <= 0;
            ldr_req    <= 0;
            streaming  <= 0;
            out_cnt    <= 0;
            w_valid    <= 0;
            w_idx      <= 0;
            w_data     <= 0;
        end else begin
            load_done <= 0;
            w_valid   <= 0;

            // 发起请求
            if (load_start && !loading && !ldr_req && !streaming) begin
                ldr_req  <= 1'b1;
                recv_cnt <= 0;
            end

            // 获得 grant 开始加载
            if (ldr_grant && ldr_req) begin
                ldr_req <= 0;
                loading <= 1;
            end

            // 接收9个tap
            if (ldr_valid && loading) begin
                tap_buf[recv_cnt] <= ldr_data;
                recv_cnt <= recv_cnt + 1;
            end

            if (ldr_done_sig && loading) begin
                loading   <= 0;
                load_done <= 1;

                // 开始 streaming 输出9个tap
                streaming <= 1;
                out_cnt   <= 0;
            end

            // streaming：逐拍输出9个tap
            if (streaming) begin
                w_valid <= 1;
                w_idx   <= out_cnt;
                w_data  <= tap_buf[out_cnt];

                if (out_cnt == 8) begin
                    streaming <= 0;
                end else begin
                    out_cnt <= out_cnt + 1;
                end
            end
        end
    end
endmodule