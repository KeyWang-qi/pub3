`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:27:39
// Design Name: 
// Module Name: bias_rom_unified
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
module bias_rom_unified #(
    parameter integer LANES     = 32,
    parameter integer TOTAL_BIAS = 11945,
    // keep your original parameter name to avoid touching other code
    parameter  BIAS_MEMH  = "D:/NoC/mycode/mobilenet_acc3/data/bias/all_bias_i32.mem"
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [11:0]              base_addr_in,
    input  wire [6:0]               block_idx,
    input  wire                     rd_en,

    output wire [LANES*32-1:0]      bias_out,
    output wire                     bias_valid
);

    localparam integer ADDR_W = $clog2(TOTAL_BIAS);

    // base + block_idx*LANES  (matches your old read_addr)
    wire [ADDR_W-1:0] addr_base =
        { {(ADDR_W-12){1'b0}}, base_addr_in } + ( { {(ADDR_W-7){1'b0}}, block_idx } * LANES );

    // XPM outputs arrive 1 cycle after rd_en
    reg rd_en_d1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rd_en_d1 <= 1'b0;
        else        rd_en_d1 <= rd_en;
    end
    assign bias_valid = rd_en_d1;

    // One SPROM per lane to get LANES parallel reads in one cycle (after latency)
    wire [31:0] lane_bias [0:LANES-1];

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : GEN_BIAS_LANES
            wire [ADDR_W-1:0] addr_i = addr_base + i[ADDR_W-1:0];

            xpm_memory_sprom #(
                .ADDR_WIDTH_A        (ADDR_W),
                .READ_DATA_WIDTH_A   (32),
                .MEMORY_SIZE         (32 * TOTAL_BIAS),
                .READ_LATENCY_A      (1),
                .MEMORY_PRIMITIVE    ("distributed"),
                .MEMORY_INIT_FILE    (BIAS_MEMH),
                .MEMORY_INIT_PARAM   (""),
                .USE_MEM_INIT        (1),
                .ECC_MODE            ("no_ecc"),
                .WAKEUP_TIME         ("disable_sleep"),
                .AUTO_SLEEP_TIME     (0),
                .MESSAGE_CONTROL     (0)
            ) u_bias_sprom (
                .clka   (clk),
                .ena    (rd_en),
                .addra  (addr_i),
                .douta  (lane_bias[i]),
                .regcea (1'b1),
                .rsta   (1'b0)
            );
            // Keep your original packing:
            // bias_out[i*32 +: 32] = bias[addr_base + i]
            assign bias_out[i*32 +: 32] = lane_bias[i];
        end
    endgenerate

endmodule
