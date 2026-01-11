`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/22 16:00:24
// Design Name: 
// Module Name: global_avg_pool
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
module global_avg_pool #(
    parameter integer CHANNELS   = 1024,
    parameter integer POOL_SIZE  = 7,       // 7x7 -> 1x1
    parameter integer DATA_W     = 8,
    parameter integer ACC_W      = 32,
    parameter integer LANES      = 16       // 每次处理16个通道
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    
    // 从feature buffer读取
    output reg         feat_rd_en,
    output reg  [15:0] feat_rd_local_addr,
    input  wire [127:0] feat_rd_data,      // 16 x int8
    input  wire        feat_rd_valid,
    
    // 写入feature buffer
    output reg         feat_wr_en,
    output reg  [15:0] feat_wr_local_addr,
    output reg [127:0] feat_wr_data,       // 16 x int8
    
    output reg  done
);

    // 7x7 = 49 像素
    // 均值计算: sum/49 ≈ (sum * 1336) >> 16 (更精确的近似)
    localparam integer PIXELS = POOL_SIZE * POOL_SIZE;  // 49
    localparam signed [31:0] DIV_MULT = 32'sd1336;      // 65536/49 ≈ 1337
    localparam integer DIV_SHIFT = 16;
    
    // 通道分块数:  1024/16 = 64
    localparam integer CH_TILES = (CHANNELS + LANES - 1) / LANES;
    
    reg [2:0] state;
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;
    localparam S_ACC     = 3'd2;
    localparam S_COMPUTE = 3'd3;
    localparam S_WRITE   = 3'd4;
    localparam S_NEXT    = 3'd5;
    localparam S_DONE    = 3'd6;
    
    reg [6:0]  ch_tile_idx;   // 0-63
    reg [5:0]  pixel_idx;     // 0-48
    reg [2:0]  row_idx;       // 0-6
    reg [2:0]  col_idx;       // 0-6
    
    // 16通道累加器
    reg signed [ACC_W-1:0] accum [0: LANES-1];
    
    // 输出寄存器
    reg signed [DATA_W-1:0] out_data [0:LANES-1];
    
    integer i;
    reg signed [63:0] div_temp;
    
    // 计算读取地址:  对于7x7特征图, 每行存储在连续地址
    // 地址 = ch_tile_idx * 49 + row_idx * 7 + col_idx
    wire [15:0] rd_addr = ch_tile_idx * PIXELS + row_idx * POOL_SIZE + col_idx;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            ch_tile_idx <= 0;
            pixel_idx <= 0;
            row_idx <= 0;
            col_idx <= 0;
            feat_rd_en <= 0;
            feat_wr_en <= 0;
            done <= 0;
            for (i = 0; i < LANES; i = i + 1)
                accum[i] <= 0;
        end else begin
            feat_wr_en <= 0;
            
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        ch_tile_idx <= 0;
                        pixel_idx <= 0;
                        row_idx <= 0;
                        col_idx <= 0;
                        for (i = 0; i < LANES; i = i + 1)
                            accum[i] <= 0;
                        state <= S_LOAD;
                    end
                end
                
                S_LOAD: begin
                    // 发起读请求
                    feat_rd_en <= 1;
                    feat_rd_local_addr <= rd_addr;
                    state <= S_ACC;
                end
                
                S_ACC: begin
                    feat_rd_en <= 0;
                    if (feat_rd_valid) begin
                        // 累加16个通道的值
                        for (i = 0; i < LANES; i = i + 1) begin
                            accum[i] <= accum[i] + $signed(feat_rd_data[i*DATA_W +: DATA_W]);
                        end
                        
                        // 更新像素索引
                        if (col_idx == POOL_SIZE - 1) begin
                            col_idx <= 0;
                            if (row_idx == POOL_SIZE - 1) begin
                                row_idx <= 0;
                                state <= S_COMPUTE;  // 一个通道块完成
                            end else begin
                                row_idx <= row_idx + 1;
                                state <= S_LOAD;
                            end
                        end else begin
                            col_idx <= col_idx + 1;
                            state <= S_LOAD;
                        end
                    end
                end
                
                S_COMPUTE: begin
                    // 计算均值:  avg = sum / 49 ≈ (sum * 1336) >> 16
                    for (i = 0; i < LANES; i = i + 1) begin
                        div_temp = (accum[i] * DIV_MULT) >>> DIV_SHIFT;
                        // 饱和到int8 [-128, 127]
                        if (div_temp > 127)
                            out_data[i] <= 8'sd127;
                        else if (div_temp < -128)
                            out_data[i] <= -8'sd128;
                        else
                            out_data[i] <= div_temp[7:0];
                    end
                    state <= S_WRITE;
                end
                
                S_WRITE: begin
                    // 写入结果 (1x1特征图, 每个通道块一个地址)
                    feat_wr_en <= 1;
                    feat_wr_local_addr <= ch_tile_idx;  // 输出地址 0-63
                    // 打包16个int8
                    for (i = 0; i < LANES; i = i + 1) begin
                        feat_wr_data[i*DATA_W +: DATA_W] <= out_data[i];
                    end
                    state <= S_NEXT;
                end
                
                S_NEXT:  begin
                    feat_wr_en <= 0;
                    if (ch_tile_idx == CH_TILES - 1) begin
                        state <= S_DONE;
                    end else begin
                        ch_tile_idx <= ch_tile_idx + 1;
                        row_idx <= 0;
                        col_idx <= 0;
                        for (i = 0; i < LANES; i = i + 1)
                            accum[i] <= 0;
                        state <= S_LOAD;
                    end
                end
                
                S_DONE: begin
                    done <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
