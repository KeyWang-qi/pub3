`timescale 1ns / 1ps
`include "mobilenet_defines.vh" 
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/01/07 15:03:50
// Design Name: 
// Module Name: pw_scheduler
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
`timescale 1ns / 1ps
`include "mobilenet_defines.vh" 
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pw_scheduler
// 优化版本：
//   - 优化1:  流水线FSM优化
//   - 优化3A: 8路并行Capture
//   - 优化A: 64个128bit Weight Buffer
//   - 优化B: Feature Pipeline预取 ? NEW
// 预期性能：251,350 cycles → 120,000 cycles (-52%) → 105,000 cycles (-58%)
//////////////////////////////////////////////////////////////////////////////////
module pw_scheduler#(
    parameter NUM_ROWS = `PE_ROWS,
    parameter NUM_COLS = `PE_COLS,
    parameter A_BITS   = `DATA_W,
    parameter W_BITS   = `WEIGHT_W,
    parameter ACC_BITS = `ACC_W,
    parameter ADDR_W   = 19,

    parameter integer FAST_SIM_EN = 1,
    parameter integer FAST_COUT_SUBSAMPLE = 16,
    parameter integer FAST_PX_SUBSAMPLE   = 16,
    parameter integer PREFETCH_TIMEOUT = 5000
)(
    input  wire CLK,
    input  wire RESET,
    input  wire start,
    output reg  done,

    input  wire [10:0] cin,
    input  wire [10:0] cout,
    input  wire [7:0]  img_w,
    input  wire [7:0]  img_h,
    input  wire [ADDR_W-1:0] w_base_in,

    output reg  weight_req,
    input  wire weight_grant,
    output reg  [ADDR_W-1:0] weight_base,
    output reg  [10:0] weight_count,
    input  wire weight_valid,
    input  wire [127:0] weight_data,
    input  wire weight_done,

    output reg  feat_rd_en,
    output reg  [15:0] feat_rd_addr,
    input  wire [127:0] feat_rd_data,
    input  wire feat_rd_valid,

    output reg  arr_W_EN,
    output reg  [NUM_COLS*W_BITS-1:0] in_weight_above,
    output reg  [NUM_ROWS*A_BITS-1:0] active_left,
    input  wire [NUM_COLS*ACC_BITS-1:0] out_sum_final,

    output reg  y_valid,
    output reg  [NUM_COLS*ACC_BITS-1:0] y_data,
    output reg  y_tile_sel
);
    // ============================================================
    // Parameters and wire declarations
    // ============================================================
    wire [10:0] cin_tiles;
    wire [10:0] cout_tiles;
    wire [15:0] total_px;
    
    assign cin_tiles  = (cin + 31) >> 5;
    assign cout_tiles = (cout + 31) >> 5;
    assign total_px   = img_w * img_h;

    localparam integer PE_LAT = NUM_ROWS - 1;
    localparam integer WEIGHT_REUSE_COUNT = 32;

    reg [31:0] work_left;
    wire [31:0] work_full;
    assign work_full = total_px * cin_tiles * cout_tiles;

    // FAST simulation
    wire [15:0] sampled_px;
    wire [10:0] sampled_cout_tiles;
    wire [31:0] work_sampled;
    wire [31:0] work_init;
    
    localparam MIN_WORK = 64;
    
    assign sampled_px = (FAST_SIM_EN != 0 && FAST_PX_SUBSAMPLE > 1) 
                        ? ((total_px + FAST_PX_SUBSAMPLE - 1) / FAST_PX_SUBSAMPLE)
                        :     total_px;

    assign sampled_cout_tiles = (FAST_SIM_EN != 0 && FAST_COUT_SUBSAMPLE > 1)
                                ? ((cout_tiles + FAST_COUT_SUBSAMPLE - 1) / FAST_COUT_SUBSAMPLE)
                                :  cout_tiles;

    assign work_sampled = sampled_px * cin_tiles * sampled_cout_tiles;

    assign work_init = (FAST_SIM_EN != 0) 
                       ? ((work_sampled < MIN_WORK) ? MIN_WORK : work_sampled)
                       : work_full;

    // ============================================================
    // ? 优化B: Feature Pipeline 寄存器
    // ============================================================
    reg [127:0] feat_pipe_stage1;       // 第一级：低128bit
    reg [127:0] feat_pipe_stage2;       // 第二级：高128bit
    reg [255:0] feat_pipe_stage3;       // 第三级：完整256bit
    reg         feat_pipe_valid1;       // stage1有效
    reg         feat_pipe_valid2;       // stage2有效
    reg         feat_pipe_valid3;       // stage3有效（可用于PE）
    
    reg [15:0]  feat_prefetch_addr_base;   // 预取地址基址
    reg         feat_prefetch_active;      // 预取激活标志
    reg [1:0]   feat_prefetch_phase;       // 预取阶段：0=idle, 1=req_low, 2=wait_low, 3=req_high
    
    wire [15:0] next_act_addr_base;        // 下一个activation的地址
    wire        should_prefetch;           // 是否应该预取
    
    // 计算下一个activation的地址
    wire [15:0] px_next_for_prefetch;
    wire [10:0] cin_next_for_prefetch;
    // Indices
    reg [15:0] px_idx;
    reg [10:0] cin_idx;
    reg [10:0] cout_idx;

    wire [15:0] px_next;
    wire [10:0] cin_next;
    wire [10:0] cout_next;
    
        reg pe_active;
    reg [5:0] pe_cycle_cnt;

    
    assign px_next_for_prefetch = (px_idx + 1 < total_px) ? (px_idx + 1) : 16'd0;
    assign cin_next_for_prefetch = (cin_idx + 1 < cin_tiles) ? (cin_idx + 1) : 11'd0;
    
    assign next_act_addr_base = (px_next_for_prefetch * cin_tiles * 2) + (cin_next_for_prefetch * 2);
    
    // 预取条件：PE正在计算 且 pipeline未满 且 未在预取中
    assign should_prefetch = pe_active && 
                            ! feat_pipe_valid3 && 
                            !feat_prefetch_active && 
                            (work_left > 1);  // 确保还有下一个iteration

    // ============================================================
    // Activation double buffer
    // ============================================================
    reg [NUM_ROWS*A_BITS-1:0] act_buf [0:1];
    reg act_buf_valid [0:1];
    reg act_buf_sel;
    reg act_load_sel;

    // ============================================================
    // Weight double buffer (64个128bit)
    // ============================================================
    reg [127:0] weight_buf [0:1][0:63];
    reg [6:0]   weight_buf_cnt [0:1];
    reg         weight_buf_valid [0:1];
    reg         weight_buf_sel;
    reg         weight_load_sel;
    reg [5:0]   weight_use_cnt [0:1];

    // Pipeline control
    reg load_active;
    reg [1:0] load_act_phase;
    reg [127:0] load_act_low;
    reg load_weight_active;


    // Capture控制（4阶段）
    reg capture_active;
    reg [1:0] capture_phase;


    
    assign px_next   = (px_idx + 1 < total_px)     ? (px_idx + 1) : 16'd0;
    assign cin_next  = (cin_idx + 1 < cin_tiles)   ? (cin_idx + 1) : 11'd0;
    assign cout_next = (cout_idx + 1 < cout_tiles) ? (cout_idx + 1) : 11'd0;

    // Accumulator
    reg signed [ACC_BITS-1:0] psum [0:NUM_COLS-1];

    // FSM状态定义
    localparam S_IDLE        = 3'd0;
    localparam S_WAIT_FIRST  = 3'd1;
    localparam S_PREFETCH    = 3'd2;
    localparam S_COMPUTE     = 3'd3;
    localparam S_OUTPUT_WAIT = 3'd4;
    localparam S_DONE        = 3'd5;
    reg [2:0] state;

    reg [31:0] prefetch_wait_cnt;

    integer i;

    // Address calculations
    wire [15:0] act_addr_base;
    wire [ADDR_W-1:0] weight_addr_base;
    wire [10:0] cout_idx_div32;
    
    assign act_addr_base = (px_idx * cin_tiles * 2) + (cin_idx * 2);
    assign cout_idx_div32 = cout_idx >> 5;
    assign weight_addr_base = w_base_in + (cout_idx_div32 * cin_tiles + cin_idx) * 64;

    // Weight offset calculation
    wire [5:0] weight_offset;
    wire [5:0] weight_offset_p1;
    
    assign weight_offset = {1'b0, weight_use_cnt[weight_buf_sel]} << 1;
    assign weight_offset_p1 = weight_offset + 6'd1;

    wire weight_buffer_exhausted;
    assign weight_buffer_exhausted = (weight_use_cnt[weight_buf_sel] >= 31);

    // 提前预取逻辑
    wire weight_buffer_near_exhausted;
    assign weight_buffer_near_exhausted = (weight_use_cnt[weight_buf_sel] >= 24);

    // ============================================================
    // Main FSM
    // ============================================================
    always @(posedge CLK or negedge RESET) begin
        if (! RESET) begin
            state <= S_IDLE;
            done <= 1'b0;
            y_valid <= 1'b0;
            y_tile_sel <= 1'b0;

            arr_W_EN <= 1'b0;
            in_weight_above <= {(NUM_COLS*W_BITS){1'b0}};
            active_left <= {(NUM_ROWS*A_BITS){1'b0}};

            weight_req <= 1'b0;
            weight_base <= {ADDR_W{1'b0}};
            weight_count <= 11'd0;

            feat_rd_en <= 1'b0;
            feat_rd_addr <= 16'd0;

            load_active <= 1'b0;
            load_act_phase <= 2'd0;
            load_act_low <= 128'd0;

            load_weight_active <= 1'b0;

            pe_active <= 1'b0;
            pe_cycle_cnt <= 6'd0;

            capture_active <= 1'b0;
            capture_phase <= 2'd0;

            px_idx <= 16'd0;
            cin_idx <= 11'd0;
            cout_idx <= 11'd0;

            act_buf_sel <= 1'b0;
            act_load_sel <= 1'b0;
            weight_buf_sel <= 1'b0;
            weight_load_sel <= 1'b0;

            work_left <= 32'd0;
            prefetch_wait_cnt <= 32'd0;

            // ? Feature Pipeline reset
            feat_pipe_stage1 <= 128'd0;
            feat_pipe_stage2 <= 128'd0;
            feat_pipe_stage3 <= 256'd0;
            feat_pipe_valid1 <= 1'b0;
            feat_pipe_valid2 <= 1'b0;
            feat_pipe_valid3 <= 1'b0;
            feat_prefetch_addr_base <= 16'd0;
            feat_prefetch_active <= 1'b0;
            feat_prefetch_phase <= 2'd0;

            for (i=0;i<NUM_COLS;i=i+1) psum[i] <= 0;
            for (i=0;i<2;i=i+1) begin
                act_buf_valid[i] <= 1'b0;
                weight_buf_valid[i] <= 1'b0;
                weight_buf_cnt[i] <= 0;
                weight_use_cnt[i] <= 0;
            end
        end else begin
            done <= 1'b0;
            y_valid <= 1'b0;
            arr_W_EN <= 1'b0;

            // ? ========== Feature Pipeline 预取逻辑 ==========
            // 自动预取：在PE计算时预取下一个activation
            if (should_prefetch && ! load_active) begin
                feat_prefetch_active <= 1'b1;
                feat_prefetch_addr_base <= next_act_addr_base;
                feat_prefetch_phase <= 2'd1;
            end
            
            // Pipeline Stage流转
            if (feat_prefetch_active) begin
                case (feat_prefetch_phase)
                    2'd1: begin
                        // 发起低128bit读取
                        if (! load_active) begin  // 确保不与load_active冲??
                            feat_rd_en <= 1'b1;
                            feat_rd_addr <= feat_prefetch_addr_base;
                            feat_prefetch_phase <= 2'd2;
                        end
                    end
                    
                    2'd2: begin
                        feat_rd_en <= 1'b0;
                        if (feat_rd_valid) begin
                            // 接收低128bit，立即发起高128bit读取
                            feat_pipe_stage1 <= feat_rd_data;
                            feat_pipe_valid1 <= 1'b1;
                            
                            feat_rd_en <= 1'b1;
                            feat_rd_addr <= feat_prefetch_addr_base + 16'd1;
                            feat_prefetch_phase <= 2'd3;
                        end
                    end
                    
                    2'd3: begin
                        feat_rd_en <= 1'b0;
                        if (feat_rd_valid) begin
                            // 接收高128bit，拼接完整256bit
                            feat_pipe_stage2 <= feat_rd_data;
                            feat_pipe_valid2 <= 1'b1;
                            
                            // 立即推进到stage3
                            feat_pipe_stage3 <= {feat_rd_data, feat_pipe_stage1};
                            feat_pipe_valid3 <= 1'b1;
                            
                            feat_prefetch_phase <= 2'd0;
                            feat_prefetch_active <= 1'b0;
                        end
                    end
                    
                    default: begin
                        feat_prefetch_phase <= 2'd0;
                        feat_prefetch_active <= 1'b0;
                    end
                endcase
            end

            // ? ========== 修改后的Load activation逻辑 ==========
            if (load_active) begin
                case (load_act_phase)
                    2'd0: begin
                        // 检查pipeline是否有现成的数据
                        if (feat_pipe_valid3 && ! feat_prefetch_active) begin
                            // ? 直接从pipeline取数据（零延迟）
                            act_buf[act_load_sel] <= feat_pipe_stage3;
                            act_buf_valid[act_load_sel] <= 1'b1;
                            
                            // 清除pipeline数据
                            feat_pipe_valid3 <= 1'b0;
                            feat_pipe_valid2 <= 1'b0;
                            feat_pipe_valid1 <= 1'b0;
                            
                            load_act_phase <= 2'd0;
                            load_active <= 1'b0;
                        end else begin
                            // Pipeline miss：走原来的路径
                            if (! feat_prefetch_active) begin  // 等待预取完成
                                feat_rd_en <= 1'b1;
                                feat_rd_addr <= act_addr_base;
                                load_act_phase <= 2'd1;
                            end
                        end
                    end
                    
                    2'd1: begin
                        feat_rd_en <= 1'b0;
                        if (feat_rd_valid) begin
                            load_act_low <= feat_rd_data;
                            feat_rd_en <= 1'b1;
                            feat_rd_addr <= act_addr_base + 16'd1;
                            load_act_phase <= 2'd2;
                        end
                    end
                    
                    2'd2: begin
                        feat_rd_en <= 1'b0;
                        if (feat_rd_valid) begin
                            act_buf[act_load_sel] <= {feat_rd_data, load_act_low};
                            act_buf_valid[act_load_sel] <= 1'b1;
                            load_act_phase <= 2'd0;
                            load_active <= 1'b0;
                        end
                    end
                endcase
            end

            // ===== Load weights (64 beats) =====
            if (load_weight_active) begin
                if (weight_buf_cnt[weight_load_sel] == 0 && !  weight_req) begin
                    weight_req <= 1'b1;
                    weight_base <= weight_addr_base;
                    weight_count <= 11'd64;
                end

                if (weight_grant) weight_req <= 1'b0;

                if (weight_valid) begin
                    weight_buf[weight_load_sel][weight_buf_cnt[weight_load_sel][5:0]] <= weight_data;
                    weight_buf_cnt[weight_load_sel] <= weight_buf_cnt[weight_load_sel] + 1;
                end

                if (weight_done) begin
                    weight_buf_valid[weight_load_sel] <= 1'b1;
                    weight_use_cnt[weight_load_sel] <= 0;
                    load_weight_active <= 1'b0;
                end
            end

            // ===== PE compute =====
            if (pe_active) begin
                if (pe_cycle_cnt == 0) begin
                    active_left <= act_buf[act_buf_sel];
                    
                    in_weight_above <= {
                        weight_buf[weight_buf_sel][weight_offset_p1[5:0]],
                        weight_buf[weight_buf_sel][weight_offset[5:0]]
                    };
                    
                    arr_W_EN <= 1'b1;
                end

                pe_cycle_cnt <= pe_cycle_cnt + 1;

                if (pe_cycle_cnt >= PE_LAT) begin
                    pe_active <= 1'b0;
                    capture_active <= 1'b1;
                    capture_phase <= 2'd0;

                    weight_use_cnt[weight_buf_sel] <= weight_use_cnt[weight_buf_sel] + 1;
                    
                    if (weight_buffer_exhausted) begin
                        weight_buf_valid[weight_buf_sel] <= 1'b0;
                        weight_buf_cnt[weight_buf_sel] <= 0;
                        weight_use_cnt[weight_buf_sel] <= 0;
                        weight_buf_sel <= ~weight_buf_sel;
                    end

                    act_buf_valid[act_buf_sel] <= 1'b0;
                    act_buf_sel <= ~act_buf_sel;
                end
            end

            // ===== 并行Capture（8路并行，4阶段）=====
            if (capture_active) begin
                case (capture_phase)
                    2'd0: begin
                        psum[0]  <= psum[0]  + $signed(out_sum_final[0*ACC_BITS +: ACC_BITS]);
                        psum[1]  <= psum[1]  + $signed(out_sum_final[1*ACC_BITS +:  ACC_BITS]);
                        psum[2]  <= psum[2]  + $signed(out_sum_final[2*ACC_BITS +: ACC_BITS]);
                        psum[3]  <= psum[3]  + $signed(out_sum_final[3*ACC_BITS +: ACC_BITS]);
                        psum[4]  <= psum[4]  + $signed(out_sum_final[4*ACC_BITS +: ACC_BITS]);
                        psum[5]  <= psum[5]  + $signed(out_sum_final[5*ACC_BITS +: ACC_BITS]);
                        psum[6]  <= psum[6]  + $signed(out_sum_final[6*ACC_BITS +: ACC_BITS]);
                        psum[7]  <= psum[7]  + $signed(out_sum_final[7*ACC_BITS +: ACC_BITS]);
                        capture_phase <= 2'd1;
                    end
                    2'd1: begin
                        psum[8]  <= psum[8]  + $signed(out_sum_final[8*ACC_BITS +: ACC_BITS]);
                        psum[9]  <= psum[9]  + $signed(out_sum_final[9*ACC_BITS +: ACC_BITS]);
                        psum[10] <= psum[10] + $signed(out_sum_final[10*ACC_BITS +: ACC_BITS]);
                        psum[11] <= psum[11] + $signed(out_sum_final[11*ACC_BITS +: ACC_BITS]);
                        psum[12] <= psum[12] + $signed(out_sum_final[12*ACC_BITS +: ACC_BITS]);
                        psum[13] <= psum[13] + $signed(out_sum_final[13*ACC_BITS +: ACC_BITS]);
                        psum[14] <= psum[14] + $signed(out_sum_final[14*ACC_BITS +: ACC_BITS]);
                        psum[15] <= psum[15] + $signed(out_sum_final[15*ACC_BITS +:  ACC_BITS]);
                        capture_phase <= 2'd2;
                    end
                    2'd2: begin
                        psum[16] <= psum[16] + $signed(out_sum_final[16*ACC_BITS +: ACC_BITS]);
                        psum[17] <= psum[17] + $signed(out_sum_final[17*ACC_BITS +: ACC_BITS]);
                        psum[18] <= psum[18] + $signed(out_sum_final[18*ACC_BITS +: ACC_BITS]);
                        psum[19] <= psum[19] + $signed(out_sum_final[19*ACC_BITS +: ACC_BITS]);
                        psum[20] <= psum[20] + $signed(out_sum_final[20*ACC_BITS +: ACC_BITS]);
                        psum[21] <= psum[21] + $signed(out_sum_final[21*ACC_BITS +: ACC_BITS]);
                        psum[22] <= psum[22] + $signed(out_sum_final[22*ACC_BITS +: ACC_BITS]);
                        psum[23] <= psum[23] + $signed(out_sum_final[23*ACC_BITS +:  ACC_BITS]);
                        capture_phase <= 2'd3;
                    end
                    2'd3: begin
                        psum[24] <= psum[24] + $signed(out_sum_final[24*ACC_BITS +: ACC_BITS]);
                        psum[25] <= psum[25] + $signed(out_sum_final[25*ACC_BITS +: ACC_BITS]);
                        psum[26] <= psum[26] + $signed(out_sum_final[26*ACC_BITS +: ACC_BITS]);
                        psum[27] <= psum[27] + $signed(out_sum_final[27*ACC_BITS +: ACC_BITS]);
                        psum[28] <= psum[28] + $signed(out_sum_final[28*ACC_BITS +: ACC_BITS]);
                        psum[29] <= psum[29] + $signed(out_sum_final[29*ACC_BITS +: ACC_BITS]);
                        psum[30] <= psum[30] + $signed(out_sum_final[30*ACC_BITS +: ACC_BITS]);
                        psum[31] <= psum[31] + $signed(out_sum_final[31*ACC_BITS +:  ACC_BITS]);
                        capture_phase <= 2'd0;
                        capture_active <= 1'b0;
                        if (work_left != 0) work_left <= work_left - 1;
                    end
                endcase
            end

            // ===== 流水线FSM =====
            case (state)
                S_IDLE: begin
                    prefetch_wait_cnt <= 0;
                    if (start) begin
                        $display("[PW] START:    work_init=%0d (full=%0d)", work_init, work_full);
                        
                        px_idx <= 0;
                        cin_idx <= 0;
                        cout_idx <= 0;
                        work_left <= work_init;
                        
                        for (i=0;i<NUM_COLS;i=i+1) psum[i] <= 0;

                        act_buf_valid[0] <= 0;
                        act_buf_valid[1] <= 0;
                        weight_buf_valid[0] <= 0;
                        weight_buf_valid[1] <= 0;
                        weight_buf_cnt[0] <= 0;
                        weight_buf_cnt[1] <= 0;
                        weight_use_cnt[0] <= 0;
                        weight_use_cnt[1] <= 0;

                        act_buf_sel <= 0;
                        weight_buf_sel <= 0;
                        act_load_sel <= 0;
                        weight_load_sel <= 0;

                        // ? 清空pipeline
                        feat_pipe_valid1 <= 1'b0;
                        feat_pipe_valid2 <= 1'b0;
                        feat_pipe_valid3 <= 1'b0;
                        feat_prefetch_active <= 1'b0;
                        feat_prefetch_phase <= 2'd0;

                        load_active <= 1;
                        load_act_phase <= 0;
                        load_weight_active <= 1;
                        weight_req <= 0;

                        state <= S_WAIT_FIRST;
                    end
                end

                S_WAIT_FIRST: begin
                    if (act_buf_valid[0] && weight_buf_valid[0]) begin
                        pe_active <= 1;
                        pe_cycle_cnt <= 0;
                        state <= S_COMPUTE;
                    end
                end

                S_COMPUTE: begin
                    // 提前预取逻辑
                    if (pe_active && weight_buffer_near_exhausted) begin
                        if (!  weight_buf_valid[~weight_buf_sel] && ! load_weight_active) begin
                            weight_load_sel <= ~weight_buf_sel;
                            load_weight_active <= 1;
                        end
                    end
                    
                    if (!  pe_active && !  capture_active) begin
                        y_valid <= 1'b1;
                        for (i=0; i<NUM_COLS; i=i+1) begin
                            y_data[i*ACC_BITS +: ACC_BITS] <= psum[i];
                        end
                        y_tile_sel <= cout_idx[0];
                        
                        for (i=0; i<NUM_COLS; i=i+1) begin
                            psum[i] <= 0;
                        end
                        
                        if (work_left == 0) begin
                            $display("[PW] DONE:  completed %0d iterations", work_init);
                            state <= S_OUTPUT_WAIT;
                        end else begin
                            px_idx   <= px_next;
                            cin_idx  <= cin_next;
                            cout_idx <= cout_next;

                            if (weight_buffer_exhausted) begin
                                act_load_sel <= ~act_buf_sel;
                                weight_load_sel <= ~weight_buf_sel;
                                
                                if (act_buf_valid[~act_buf_sel] && weight_buf_valid[~weight_buf_sel]) begin
                                    act_buf_sel <= ~act_buf_sel;
                                    weight_buf_sel <= ~weight_buf_sel;
                                    pe_active <= 1;
                                    pe_cycle_cnt <= 0;
                                end else begin
                                    load_active <= 1;
                                    load_act_phase <= 0;
                                    load_weight_active <= 1;
                                    state <= S_PREFETCH;
                                end
                            end else begin
                                act_load_sel <= ~act_buf_sel;
                                weight_load_sel <= weight_buf_sel;
                                
                                if (act_buf_valid[~act_buf_sel]) begin
                                    act_buf_sel <= ~act_buf_sel;
                                    pe_active <= 1;
                                    pe_cycle_cnt <= 0;
                                end else begin
                                    load_active <= 1;
                                    load_act_phase <= 0;
                                    load_weight_active <= 0;
                                    state <= S_PREFETCH;
                                end
                            end
                        end
                    end
                end

                S_PREFETCH: begin
                    if (act_buf_valid[act_load_sel] && 
                        (weight_buf_valid[weight_load_sel] || !  load_weight_active)) begin
                        
                        if (load_weight_active && weight_load_sel != weight_buf_sel) begin
                            weight_buf_sel <= weight_load_sel;
                        end
                        act_buf_sel <= act_load_sel;
                        
                        pe_active <= 1;
                        pe_cycle_cnt <= 0;
                        state <= S_COMPUTE;
                    end
                end

                S_OUTPUT_WAIT: begin
                    y_valid <= 1'b0;
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
                
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule