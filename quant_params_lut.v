`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:18:59
// Design Name: 
// Module Name: quant_params_lut
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
// Module Name: quant_params_lut
// Description: MobileNetV1 所有层的量化参数查找表
//              支持28层 (Layer0-26 + FC)
//////////////////////////////////////////////////////////////////////////////////
module quant_params_lut (
    input  wire [4:0] layer_sel,
    output reg signed [31:0] mult_scalar,
    output reg        [5:0]  shift_scalar,
    output reg signed [7:0]  zp_out
);
    always @(*) begin
        case (layer_sel)
            5'd0: begin
                mult_scalar  = 32'd1499917960;
                shift_scalar = 6'd36;
                zp_out       = -8'd105;
            end
            5'd1: begin
                mult_scalar  = 32'd1254985707;
                shift_scalar = 6'd32;
                zp_out       = 8'd110;
            end
            5'd2: begin
                mult_scalar  = 32'd2090511766;
                shift_scalar = 6'd36;
                zp_out       = 8'd121;
            end
            5'd3: begin
                mult_scalar  = 32'd1729896176;
                shift_scalar = 6'd32;
                zp_out       = -8'd126;
            end
            5'd4: begin
                mult_scalar  = 32'd2081950144;
                shift_scalar = 6'd37;
                zp_out       = 8'd104;
            end
            5'd5: begin
                mult_scalar  = 32'd2080045790;
                shift_scalar = 6'd35;
                zp_out       = -8'd96;
            end
            5'd6: begin
                mult_scalar  = 32'd1890535752;
                shift_scalar = 6'd37;
                zp_out       = 8'd94;
            end
            5'd7: begin
                mult_scalar  = 32'd1151606283;
                shift_scalar = 6'd36;
                zp_out       = 8'd123;
            end
            5'd8: begin
                mult_scalar  = 32'd2089579792;
                shift_scalar = 6'd38;
                zp_out       = -8'd105;
            end
            5'd9: begin
                mult_scalar  = 32'd1410648336;
                shift_scalar = 6'd35;
                zp_out       = -8'd127;
            end
            5'd10: begin
                mult_scalar  = 32'd1767908595;
                shift_scalar = 6'd38;
                zp_out       = 8'd122;
            end
            5'd11: begin
                mult_scalar  = 32'd1850037303;
                shift_scalar = 6'd37;
                zp_out       = 8'd122;
            end
            5'd12: begin
                mult_scalar  = 32'd1260482948;
                shift_scalar = 6'd37;
                zp_out       = 8'd109;
            end
            5'd13: begin
                mult_scalar  = 32'd1269068553;
                shift_scalar = 6'd35;
                zp_out       = -8'd124;
            end
            5'd14: begin
                mult_scalar  = 32'd1456865826;
                shift_scalar = 6'd38;
                zp_out       = -8'd116;
            end
            5'd15: begin
                mult_scalar  = 32'd1464063745;
                shift_scalar = 6'd35;
                zp_out       = 8'd94;
            end
            5'd16: begin
                mult_scalar  = 32'd1364297475;
                shift_scalar = 6'd38;
                zp_out       = 8'd127;
            end
            5'd17: begin
                mult_scalar  = 32'd1948806020;
                shift_scalar = 6'd36;
                zp_out       = 8'd127;
            end
            5'd18: begin
                mult_scalar  = 32'd2136047628;
                shift_scalar = 6'd38;
                zp_out       = 8'd89;
            end
            5'd19: begin
                mult_scalar  = 32'd1671906936;
                shift_scalar = 6'd36;
                zp_out       = -8'd122;
            end
            5'd20: begin
                mult_scalar  = 32'd1327474817;
                shift_scalar = 6'd37;
                zp_out       = 8'd99;
            end
            5'd21: begin
                mult_scalar  = 32'd1330877187;
                shift_scalar = 6'd36;
                zp_out       = 8'd106;
            end
            5'd22: begin
                mult_scalar  = 32'd1497258227;
                shift_scalar = 6'd38;
                zp_out       = -8'd103;
            end
            5'd23: begin
                mult_scalar  = 32'd1076915977;
                shift_scalar = 6'd37;
                zp_out       = 8'd126;
            end
            5'd24: begin
                mult_scalar  = 32'd1124144811;
                shift_scalar = 6'd37;
                zp_out       = -8'd126;
            end
            5'd25: begin
                mult_scalar  = 32'd1083785863;
                shift_scalar = 6'd33;
                zp_out       = -8'd45;
            end
            5'd26: begin
                mult_scalar  = 32'd1240259561;
                shift_scalar = 6'd36;
                zp_out       = 8'd95;
            end
            
            // Layer 27: AP (不需要量化，但需要有效条目)
      5'd27: begin
        mult_scalar  = 32'd1;
        shift_scalar = 6'd0;
        zp_out       = 8'sd0;
      end
            5'd28: begin
                mult_scalar  = 32'd1370706446;
                shift_scalar = 6'd38;
                zp_out       = 8'd74;
            end
            default: begin mult_scalar=0; shift_scalar=0; zp_out=0; end
        endcase
    end
endmodule
