`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:19:47
// Design Name: 
// Module Name: blk_mem_gen_unified
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
module blk_mem_gen_unified #(
    parameter integer BUF_ADDR_W = 15, 
    parameter integer WIDTH      = 128, 
    parameter integer DEPTH      = 32768
)(
    input  wire                   clka,
    
    input  wire                   ena,   
    input  wire [BUF_ADDR_W-1:0]  addra, 
    output wire [WIDTH-1:0]       douta, 

    input  wire                   dma_wr_en,   
    input  wire [BUF_ADDR_W-1:0]  dma_wr_addr, 
    input  wire [WIDTH-1:0]       dma_wr_data  
);

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A        (BUF_ADDR_W),
        .ADDR_WIDTH_B        (BUF_ADDR_W),
        .WRITE_DATA_WIDTH_A  (WIDTH),
        .READ_DATA_WIDTH_B   (WIDTH),
        .BYTE_WRITE_WIDTH_A  (WIDTH),    

        .MEMORY_SIZE         (WIDTH * DEPTH), 
        .MEMORY_PRIMITIVE    ("block"),       
        
        .READ_LATENCY_B      (2),
        .WRITE_MODE_B        ("read_first"),
        .USE_MEM_INIT        (0),
        .ECC_MODE            ("no_ecc")
    ) u_weight_buf (
        // ================== Port A (Write) ==================
        .clka   (clka),
        .ena    (dma_wr_en), 
        
        .wea    (dma_wr_en),
        .addra  (dma_wr_addr),
        .dina   (dma_wr_data),

        // ================== Port B (Read) ==================
        .clkb   (clka),
        .enb    (ena), 
        .addrb  (addra),
        .doutb  (douta),

        .regceb (1'b1),
        .sleep  (1'b0)  
    );
endmodule