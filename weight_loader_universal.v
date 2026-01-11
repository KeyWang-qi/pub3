`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:25:41
// Design Name: 
// Module Name: weight_loader_universal
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
module weight_loader_universal #(
    parameter ADDR_W = 19,
    parameter DATA_W = 128,
    parameter integer RD_LAT = 2,           // 对应 BRAM Latency = 2
    parameter integer SIM_BYPASS_PRELOAD = 1,
    // [新增] 参数化 BRAM 地址位宽，默认为 15 (512KB)
    parameter integer BUF_ADDR_W = 15 
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // 控制接口
    input  wire                   start,
    input  wire [ADDR_W-1:0]      base_addr,
    input  wire [16:0]            load_count,
    output reg                    done,

    // preload handshake
    output reg                    preload_req,
    output reg  [ADDR_W-1:0]      preload_base,
    output reg  [16:0]            preload_count,
    input  wire                   preload_done,

    // weight buffer interface
    output reg                    bmg_en,
    // [修改] 使用参数化位宽 (15位)
    output reg  [BUF_ADDR_W-1:0]  bmg_addr, 
    input  wire [DATA_W-1:0]      bmg_data,

    // stream out
    output reg                    out_valid,
    output reg  [DATA_W-1:0]      out_data
);

    // ----------------------------
    // FSM states
    // ----------------------------
    reg [1:0] state;
    localparam S_IDLE    = 2'd0;
    localparam S_PRELOAD = 2'd1;
    localparam S_READ    = 2'd2;
    localparam S_WAIT    = 2'd3;

    reg [16:0] cnt;

    // ----------------------------
    // start rising edge detect
    // ----------------------------
    reg start_d;
    wire start_rise = start & ~start_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) start_d <= 1'b0;
        else        start_d <= start;
    end

    // ----------------------------
    // 仿真保护
    // ----------------------------
    wire [16:0] load_count_safe;
`ifndef SYNTHESIS
    assign load_count_safe =
        ((^load_count === 1'bX) || (load_count == 17'd0)) ? 17'd64 : load_count;
`else
    assign load_count_safe = load_count;
`endif

    // ----------------------------
    // preload_done bypass in sim
    // ----------------------------
    wire preload_done_i;
    assign preload_done_i = (SIM_BYPASS_PRELOAD != 0) ? 1'b1 : preload_done;

    // ----------------------------
    // pipeline for read latency
    // ----------------------------
    reg [RD_LAT-1:0] bmg_en_pipe;
    integer i;

    // ----------------------------
    // main FSM
    // ----------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            cnt           <= 17'd0;
            bmg_en        <= 1'b0;
            // [修改] 复位为 0
            bmg_addr      <= {BUF_ADDR_W{1'b0}}; 

            out_valid     <= 1'b0;
            out_data      <= {DATA_W{1'b0}};
            done          <= 1'b0;

            preload_req   <= 1'b0;
            preload_base  <= {ADDR_W{1'b0}};
            preload_count <= 17'd0;

            bmg_en_pipe   <= {RD_LAT{1'b0}};
        end else begin
            // defaults
            out_valid <= 1'b0;
            done      <= 1'b0;

            // shift pipeline
            bmg_en_pipe[0] <= bmg_en;
            for (i = 1; i < RD_LAT; i = i + 1)
                bmg_en_pipe[i] <= bmg_en_pipe[i-1];

            // data becomes valid after RD_LAT
            if (bmg_en_pipe[RD_LAT-1]) begin
                out_valid <= 1'b1;
                out_data  <= bmg_data;
            end

            case (state)
                S_IDLE: begin
                    bmg_en      <= 1'b0;
                    preload_req <= 1'b0;

                    if (start_rise) begin
                        preload_base  <= base_addr;
                        preload_count <= load_count_safe;
                        cnt           <= 17'd0;

                        if (SIM_BYPASS_PRELOAD != 0) begin
                            // ? 仿真 bypass：直接进入 READ
                            bmg_en   <= 1'b1;
                            bmg_addr <= {BUF_ADDR_W{1'b0}};
                            state    <= S_READ;
                        end else begin
                            // ? 真实 preload
                            preload_req <= 1'b1;
                            state       <= S_PRELOAD;
                        end
                    end
                end

                // --------------------------------
                // PRELOAD
                // --------------------------------
                S_PRELOAD: begin
                    bmg_en <= 1'b0;

                    if (preload_done_i) begin
                        preload_req <= 1'b0;
                        cnt         <= 17'd0;
                        bmg_en      <= 1'b1;
                        bmg_addr    <= {BUF_ADDR_W{1'b0}};
                        state       <= S_READ;
                    end
                end

                // --------------------------------
                // READ: 连续输出 load_count_safe 次
                // --------------------------------
                S_READ: begin
                    if (cnt < load_count_safe - 1) begin
                        bmg_en   <= 1'b1;
                        bmg_addr <= bmg_addr + 1'b1;
                        cnt      <= cnt + 1'b1;
                    end else begin
                        bmg_en <= 1'b0;
                        state  <= S_WAIT;
                    end
                end

                // --------------------------------
                // WAIT: drain pipeline
                // --------------------------------
                S_WAIT: begin
                    if (bmg_en_pipe[RD_LAT-1] == 1'b0) begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule