`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/22 16:15:27
// Design Name: 
// Module Name: fc_layer
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

// Fully Connected Layer for MobileNetV1 Layer 28
// Input:  1x1x1024 (from Global Average Pooling)
// Output: 1000 class logits
// 权重矩阵: 1024 x 1000 = 1,024,000 个 int8 权重
//////////////////////////////////////////////////////////////////////////////////
module fc_layer #(
    parameter integer IN_FEATURES  = 1024,
    parameter integer OUT_CLASSES  = 1000,
    parameter integer DATA_W       = 8,
    parameter integer ACC_W        = 32,
    parameter integer LANES        = 16,
    parameter integer ADDR_W       = 19,
    parameter integer FAST_NO_MUL = 0
)(

    input  wire clk,
    input  wire rst_n,
    input  wire start,
    
    input  wire [ADDR_W-1:0] w_base,
    input  wire [11:0]       b_base,

    input  wire signed [15:0] quant_M,
    input  wire [5:0]         quant_s,
    input  wire signed [7:0]  quant_zp,
    
    output reg         weight_req,
    output reg  [ADDR_W-1:0] weight_base,
    output reg  [10:0] weight_count,
    input  wire        weight_grant,
    input  wire        weight_valid,
    input  wire [127:0] weight_data,
    input  wire        weight_done,
    
    input  wire [511:0] bias_vec,
    input  wire        bias_valid,
    output reg  [6:0]  bias_block_idx,
    output reg         bias_rd_en,
    
    // Feature buffer? (?AP)
    output reg         feat_rd_en,
    output reg  [15:0] feat_rd_local_addr,
    input  wire [127:0] feat_rd_data,
    input  wire        feat_rd_valid,
    
    // 
    output reg         out_valid,
    output reg  [10:0] out_class_idx,
    output reg  signed [DATA_W-1:0] out_logit,
    output reg         done
);

    // ??
    reg [3:0] state;
    localparam S_IDLE      = 4'd0;
    localparam S_LOAD_IN   = 4'd1;   // 
    localparam S_LOAD_W    = 4'd2;   // ?
    localparam S_MAC       = 4'd3;   // ?
    localparam S_BIAS      = 4'd4;   // bias
    localparam S_QUANT     = 4'd5;   // 
    localparam S_OUTPUT    = 4'd6;   // 
    localparam S_NEXT      = 4'd7;   // ?
    localparam S_DONE      = 4'd8;
    
    //  (1024 x int8 = 64 x 16)
    reg signed [DATA_W-1:0] input_feat [0:IN_FEATURES-1];
    
    // tile:  ?μ16
    localparam integer OUT_TILES = (OUT_CLASSES + LANES - 1) / LANES;  // ceil(1000/16) = 63
    
    reg [6:0]  out_tile_idx;  // 0-62
    reg [10:0] in_idx;        // 0-1023
    reg [6:0]  in_tile_idx;   // 0-63 (?)
    
    // ? (16?)
    reg signed [ACC_W-1:0] accum [0:LANES-1];
    
    // ??
    reg signed [DATA_W-1:0] weight_cache [0:LANES-1];
    
    integer i, j;
    reg signed [63:0] quant_temp;
    reg signed [31:0] bias_temp [0:LANES-1];
    
    always @(posedge clk or negedge rst_n) begin
        if (! rst_n) begin
            state <= S_IDLE;
            out_tile_idx <= 0;
            in_idx <= 0;
            in_tile_idx <= 0;
            weight_req <= 0;
            feat_rd_en <= 0;
            bias_rd_en <= 0;
            out_valid <= 0;
            done <= 0;
            for (i = 0; i < LANES; i = i + 1)
                accum[i] <= 0;
        end else begin
            out_valid <= 0;
            feat_rd_en <= 0;
            bias_rd_en <= 0;
            
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        out_tile_idx <= 0;
                        in_tile_idx <= 0;
                        for (i = 0; i < LANES; i = i + 1)
                            accum[i] <= 0;
                        state <= S_LOAD_IN;
                    end
                end
                
                //  (64ζ?, ?16)
                S_LOAD_IN:  begin
                    feat_rd_en <= 1;
                    feat_rd_local_addr <= in_tile_idx;
                    if (feat_rd_valid) begin
                        // 洢16
                        for (i = 0; i < LANES; i = i + 1) begin
                            input_feat[in_tile_idx * LANES + i] <= feat_rd_data[i*DATA_W +: DATA_W];
                        end
                        if (in_tile_idx == IN_FEATURES/LANES - 1) begin
                            in_tile_idx <= 0;
                            in_idx <= 0;
                            state <= S_LOAD_W;
                            weight_req <= 1;
                            // ???:  w_base + out_tile_idx * IN_FEATURES
                            weight_base <= w_base + out_tile_idx * (IN_FEATURES / LANES);
                            weight_count <= IN_FEATURES / LANES;  // 64128λword
                        end else begin
                            in_tile_idx <= in_tile_idx + 1;
                        end
                    end
                end
                
                // ??
                S_LOAD_W: begin
                    if (weight_grant) begin
                        weight_req <= 0;
                    end
                    if (weight_valid) begin
                        // 16?
                        for (i = 0; i < LANES; i = i + 1) begin
                            weight_cache[i] <= weight_data[i*DATA_W +: DATA_W];
                        end
                        state <= S_MAC;
                    end
                end
                
                // ?
                S_MAC: begin
                    // accum[j] += input_feat[in_idx] * weight[j] 
                    // :  ??1024
                    for (i = 0; i < LANES; i = i + 1) begin
                        accum[i] <= accum[i] +
                            ( (FAST_NO_MUL != 0) ?
                                // FAST: avoid multiplier, keep adder activity
                                $signed({{(ACC_W-DATA_W){input_feat[in_idx][DATA_W-1]}}, input_feat[in_idx]})
                              :
                                $signed(input_feat[in_idx]) * $signed(weight_cache[i])
                            );
                    end
                    
                    if (in_idx == IN_FEATURES - 1) begin
                        in_idx <= 0;
                        state <= S_BIAS;
                        bias_rd_en <= 1;
                        bias_block_idx <= out_tile_idx;
                    end else begin
                        in_idx <= in_idx + 1;
                        state <= S_LOAD_W;
                        weight_req <= 1;
                    end
                end
                
                // ?bias
                S_BIAS: begin
                    bias_rd_en <= 0;
                    if (bias_valid) begin
                        for (i = 0; i < LANES; i = i + 1) begin
                            bias_temp[i] <= bias_vec[i*32 +: 32];
                            accum[i] <= accum[i] + $signed(bias_vec[i*32 +: 32]);
                        end
                        state <= S_QUANT;
                    end
                end
                
                // 
                S_QUANT: begin
                    state <= S_OUTPUT;
                end
                
                // 
                S_OUTPUT: begin
                    // 16logits
                    for (i = 0; i < LANES; i = i + 1) begin
                        if (out_tile_idx * LANES + i < OUT_CLASSES) begin
                            // :  (acc * M) >> s + zp
                            quant_temp = (accum[i] * quant_M) >>> quant_s;
                            quant_temp = quant_temp + quant_zp;
                            // 
                            if (quant_temp > 127)
                                out_logit <= 8'sd127;
                            else if (quant_temp < -128)
                                out_logit <= -8'sd128;
                            else
                                out_logit <= quant_temp[7:0];
                            out_class_idx <= out_tile_idx * LANES + i;
                            out_valid <= 1;
                        end
                    end
                    state <= S_NEXT;
                end
                
                S_NEXT: begin
                    if (out_tile_idx == OUT_TILES - 1) begin
                        state <= S_DONE;
                    end else begin
                        out_tile_idx <= out_tile_idx + 1;
                        for (i = 0; i < LANES; i = i + 1)
                            accum[i] <= 0;
                        in_idx <= 0;
                        state <= S_LOAD_W;
                        weight_req <= 1;
                        weight_base <= w_base + (out_tile_idx + 1) * (IN_FEATURES / LANES);
                        weight_count <= IN_FEATURES / LANES;
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

