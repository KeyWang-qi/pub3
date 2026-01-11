`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:32:16
// Design Name: 
// Module Name: PE_single_weight
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
module PE_single_weight(
    // System interface
    input wire CLK,
    input wire RESET, 
    input wire EN,
    input wire W_EN,
    
    // PE row interface
    input wire signed [7:0] active_left,
    output reg signed [7:0] active_right,
    
    // PE column interface
    input wire signed [31:0] in_sum,

    (* use_dsp = "yes" *) 
    output reg signed [31:0] out_sum,
    
    // Weight flow interface
    input wire signed [7:0] in_weight_above,
    output reg signed [7:0] out_weight_below
);

    reg signed [7:0] weight;
    always @(posedge CLK) begin
        if (RESET) begin // 假设 RESET 高电平有效，如果是低有效改为 !RESET
            out_sum          <= 32'sd0;
            active_right     <= 8'sd0;
            out_weight_below <= 8'sd0;
            weight           <= 8'sd0;
        end
        else if (EN) begin
            active_right <= active_left;
            // 2. 权重加载逻辑 (解耦)
            if (W_EN) begin
                weight           <= in_weight_above;
                out_weight_below <= in_weight_above;
            end
            if (!W_EN) begin
                out_sum <= $signed(in_sum) + ($signed(active_left) * $signed(weight));
            end
        end
    end
endmodule