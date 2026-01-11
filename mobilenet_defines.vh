`ifndef MOBILENET_DEFINES_VH
`define MOBILENET_DEFINES_VH

// ============================================================
// MobileNetV1 全局参数定义 (32×32 PE阵列)
// ============================================================

// -------------------- 数据位宽 --------------------
`define DATA_W      8      // INT8
`define ACC_W       32     // 累加器位宽
`define WEIGHT_W    8      // 权重位宽
`define PSUM_W      18     // 部分和位宽
`define PROD_W      16     // 乘积位宽

// -------------------- PE阵列配置 (32×32) --------------------
`define PE_ROWS     32
`define PE_COLS     32
`define UNIT_NUM    32
`define LANES       32

// -------------------- MobileNetV1 层配置 --------------------
`define TOTAL_LAYERS              29     // Layer 0-28 (共29层)
`define MAX_LAYER_ID              28
`define TOTAL_CONV_DW_PW_LAYERS   27

// -------------------- 图像尺寸配置 --------------------
`define MAX_IMG_W     224
`define MAX_IMG_H     224
`define MAX_CHANNELS  1024
`define MAX_OUT_W     112
`define MAX_OUT_H     112

// -------------------- 特征图缓冲区深度 --------------------
// (112*112*64 / 16 = 50176 words)
`define FEATURE_BUF_DEPTH  50176

// -------------------- 权重存储基地址 --------------------
`define L0_WEIGHT_BASE   9'd0     // Layer0 从 0 开始
`define L2_WEIGHT_BASE   9'd64    // Layer2 从 64 开始

// -------------------- 数据路径 --------------------
`define DATA_PATH  "D:/NoC/mycode/mobilenet_acc2/data/"

// ============================================================
// Layer Type Definitions (3-bit)
// ============================================================
`define LAYER_TYPE_CONV   3'd0    // 第一个标准3x3卷积
`define LAYER_TYPE_DW     3'd1    // Depthwise Convolution
`define LAYER_TYPE_PW     3'd2    // Pointwise Convolution
`define LAYER_TYPE_AP     3'd3    // Global Average Pooling
`define LAYER_TYPE_FC     3'd4    // Fully Connected

// ============================================================
// Memory Parameters
// ============================================================
`define WEIGHT_ADDR_W   16
`define BIAS_ADDR_W     12
`define FEAT_ADDR_W     17

// ============================================================
// Fast Simulation Parameters
// ============================================================
`define FAST_SIM_ENABLE  0    // 默认关闭
`define FAST_SIM_NO_MAC  0    // 默认关闭

`endif  // MOBILENET_DEFINES_VH
