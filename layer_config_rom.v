`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/12/15 16:18:14
// Design Name: 
// Module Name: layer_config_rom
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

module layer_config_rom (
    input      [4:0] id,
    output reg [19:0] w_base,
    output reg [11:0] b_base,
    // 新增字段
    output reg [2:0]  layer_type,  // 0=CONV, 1=DW, 2=PW, 3=AP，4=FC
    output reg [10:0]  cin,         // 输入通道数（最大1024）
    output reg [10:0]  cout,        // 输出通道数
    output reg [7:0]  img_w,       // 特征图宽度
    output reg [7:0]  img_h,       // 特征图高度
    output reg [1:0]  stride       // 步长（1 or 2）
);

 // Layer type definitions
  localparam [2:0] TYPE_CONV = 3'd0;  // 第一层3x3卷积
  localparam [2:0] TYPE_DW   = 3'd1;  // Depthwise卷积
  localparam [2:0] TYPE_PW   = 3'd2;  // Pointwise卷积
  localparam [2:0] TYPE_AP   = 3'd3;  // Average Pooling (Global)
  localparam [2:0] TYPE_FC   = 3'd4;  // Fully Connected


  always @(*) begin
    case(id)
      // Layer 0: CONV 3x3, 3->32, 224x224, stride=2
      5'd0: begin
        w_base = 20'd0; b_base = 12'd0;
        layer_type = 3'd0; cin = 10'd3; cout = 10'd32;
        img_w = 8'd224; img_h = 8'd224; stride = 2'd2;
      end
      
      // Layer 1: DW 3x3, 32, 112x112, stride=1
      5'd1: begin
        w_base = 20'd54; b_base = 12'd2;
        layer_type = 3'd1; cin = 10'd32; cout = 10'd32;
        img_w = 8'd112; img_h = 8'd112; stride = 2'd1;
      end
      
      // Layer 2: PW 1x1, 32->64, 112x112
      5'd2: begin
        w_base = 20'd72; b_base = 12'd4;
        layer_type = 3'd2; cin = 10'd32; cout = 10'd64;
        img_w = 8'd112; img_h = 8'd112; stride = 2'd1;
      end
      
      // Layer 3: DW 3x3, 64, 112x112, stride=2
      5'd3: begin
        w_base = 20'd200; b_base = 12'd8;
        layer_type = 3'd1; cin = 10'd64; cout = 10'd64;
        img_w = 8'd112; img_h = 8'd112; stride = 2'd2;
      end
      
      // Layer 4: PW 1x1, 64->128, 56x56
      5'd4: begin
        w_base = 20'd236; b_base = 12'd12;
        layer_type = 3'd2; cin = 10'd64; cout = 10'd128;
        img_w = 8'd56; img_h = 8'd56; stride = 2'd1;
      end
      
      // Layer 5: DW 3x3, 128, 56x56, stride=1
      5'd5: begin
        w_base = 20'd748; b_base = 12'd20;
        layer_type = 3'd1; cin = 10'd128; cout = 10'd128;
        img_w = 8'd56; img_h = 8'd56; stride = 2'd1;
      end
      
      // Layer 6: PW 1x1, 128->128, 56x56
      5'd6: begin
        w_base = 20'd820; b_base = 12'd28;
        layer_type = 3'd2; cin = 10'd128; cout = 10'd128;
        img_w = 8'd56; img_h = 8'd56; stride = 2'd1;
      end
      
      // Layer 7: DW 3x3, 128, 56x56, stride=2
      5'd7: begin
        w_base = 20'd1844; b_base = 12'd36;
        layer_type = 3'd1; cin = 10'd128; cout = 10'd128;
        img_w = 8'd56; img_h = 8'd56; stride = 2'd2;
      end
      
      // Layer 8: PW 1x1, 128->256, 28x28
      5'd8: begin
        w_base = 20'd1916; b_base = 12'd44;
        layer_type = 3'd2; cin = 10'd128; cout = 10'd256;
        img_w = 8'd28; img_h = 8'd28; stride = 2'd1;
      end
      
      // Layer 9: DW 3x3, 256, 28x28, stride=1
      5'd9: begin
        w_base = 20'd3964; b_base = 12'd60;
        layer_type = 3'd1; cin = 10'd256; cout = 10'd256;
        img_w = 8'd28; img_h = 8'd28; stride = 2'd1;
      end
      
      // Layer 10: PW 1x1, 256->256, 28x28
      5'd10: begin
        w_base = 20'd4108; b_base = 12'd76;
        layer_type = 3'd2; cin = 10'd256; cout = 10'd256;
        img_w = 8'd28; img_h = 8'd28; stride = 2'd1;
      end
      
      // Layer 11: DW 3x3, 256, 28x28, stride=2
      5'd11: begin
        w_base = 20'd8204; b_base = 12'd92;
        layer_type = 3'd1; cin = 10'd256; cout = 10'd256;
        img_w = 8'd28; img_h = 8'd28; stride = 2'd2;
      end
      
      // Layer 12: PW 1x1, 256->512, 14x14
      5'd12: begin
        w_base = 20'd8348; b_base = 12'd108;
        layer_type = 3'd2; cin = 10'd256; cout = 10'd512;
        img_w = 8'd14; img_h = 8'd14; stride = 2'd1;
      end
      
      // Layers 13-22: 5 次重复 (DW 512 + PW 512->512)
      5'd13: begin w_base = 20'd16540; b_base = 12'd140; layer_type = 3'd1; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd14: begin w_base = 20'd16828; b_base = 12'd172; layer_type = 3'd2; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd15: begin w_base = 20'd33212; b_base = 12'd204; layer_type = 3'd1; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd16: begin w_base = 20'd33500; b_base = 12'd236; layer_type = 3'd2; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd17: begin w_base = 20'd49884; b_base = 12'd268; layer_type = 3'd1; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd18: begin w_base = 20'd50172; b_base = 12'd300; layer_type = 3'd2; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd19: begin w_base = 20'd66556; b_base = 12'd332; layer_type = 3'd1; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd20: begin w_base = 20'd66844; b_base = 12'd364; layer_type = 3'd2; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd21: begin w_base = 20'd83228; b_base = 12'd396; layer_type = 3'd1; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      5'd22: begin w_base = 20'd83516; b_base = 12'd428; layer_type = 3'd2; cin = 10'd512; cout = 10'd512; img_w = 8'd14; img_h = 8'd14; stride = 2'd1; end
      
      // Layer 23: DW 3x3, 512, 14x14, stride=2
      5'd23: begin
        w_base = 20'd99900; b_base = 12'd460;
        layer_type = 3'd1; cin = 10'd512; cout = 10'd512;
        img_w = 8'd14; img_h = 8'd14; stride = 2'd2;
      end
      
      // Layer 24: PW 1x1, 512->1024, 7x7
      5'd24: begin
        w_base = 20'd100188; b_base = 12'd492;
        layer_type = 3'd2; cin = 10'd512; cout = 11'd1024;
        img_w = 8'd7; img_h = 8'd7; stride = 2'd1;
      end
      
      // Layer 25: DW 3x3, 1024, 7x7, stride=1
      5'd25: begin
        w_base = 20'd132956; b_base = 12'd556;
        layer_type = 3'd1; cin = 11'd1024; cout = 11'd1024;
        img_w = 8'd7; img_h = 8'd7; stride = 2'd1;
      end
      
      // Layer 26: PW 1x1, 1024->1024, 7x7
      5'd26: begin
        w_base = 20'd133532; b_base = 12'd620;
        layer_type = 3'd2; cin = 11'd1024; cout = 11'd1024;
        img_w = 8'd7; img_h = 8'd7; stride = 2'd1;
      end
      
      // ============ 修正:  Layer 27 是 Global Average Pooling ============
      5'd27: begin
        w_base = 20'd0;      // AP不需要权重
        b_base = 12'd0;      // AP不需要bias
        layer_type = TYPE_AP;  // 3'd3 = Average Pooling
        cin  = 11'd1024;     // 输入1024通道
        cout = 11'd1024;     // 输出1024通道(通道数不变)
        img_w = 8'd7;        // 输入7x7
        img_h = 8'd7;        // 输入7x7
        stride = 2'd1;       // 输出1x1 (全局池化)
      end
      
      // ============ Layer 28: Fully Connected 1024->1000 ============
      5'd28: begin
        w_base = 20'd199068; b_base = 12'd684;
        layer_type = TYPE_FC;  // 3'd4 = Fully Connected
        cin  = 11'd1024;     // 输入1024特征
        cout = 11'd1000;     // 输出1000类
        img_w = 8'd1;        // 输入1x1 (来自AP输出)
        img_h = 8'd1;
        stride = 2'd1;
      end
      
      default: begin
        w_base = 20'd0; b_base = 12'd0;
        layer_type = 3'd0; cin = 11'd0; cout = 11'd0;
        img_w = 8'd0; img_h = 8'd0; stride = 2'd0;
      end
    endcase
  end
endmodule

