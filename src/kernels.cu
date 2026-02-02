#include <cuda_fp16.h>
#include <vector>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__global__ void trace_kernel(const T *d_input, size_t rows, size_t cols, T *d_output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned char s_mem[]; 
    T* current_data = reinterpret_cast<T*>(s_mem);
    if (idx >= rows || idx >= cols) {
        current_data[tid] = 0;
    } else {
        current_data[tid] = d_input[idx * cols + idx];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            current_data[tid] += current_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_output, current_data[0]);
    }
}

template <typename T> T trace(const std::vector<T> &h_input, size_t rows, size_t cols) {
    // TODO: Implement the trace function
    T *d_input;
    T *d_output;
    T h_output = 0;
    size_t size = rows * cols * sizeof(T);
    cudaMalloc((void **)&d_input, size);
    cudaMemcpyAsync(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_output, sizeof(T));
    cudaMemset(d_output, 0, sizeof(T));
    int threadsPerBlock = 256;
    int blocksPerGrid = (std::min(rows, cols) + threadsPerBlock - 1) / threadsPerBlock;
    trace_kernel<T><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(d_input, rows, cols, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return h_output;
}

template <int Br, int Bc, int D_HEAD, int BLOCK_SIZE>
__global__ void flash_fwd_kernel_float(const float *Q, const float *K, const float *V, float *O, int batch_size,
                                       int num_q_heads, int num_kv_heads, int seq_len_q, int seq_len_k,
                                       bool is_causal) {
    int batch_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int row_block_idx = blockIdx.z;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / gqa_ratio;
    const int q_block_start_row = row_block_idx * Br;
    const int thread_id = threadIdx.x;

    // Strides
    const long long stride_batch_q = (long long)seq_len_q * num_q_heads * D_HEAD;
    const long long stride_seq_q = (long long)num_q_heads * D_HEAD;
    const long long stride_head_q = (long long)D_HEAD;
    const long long stride_batch_kv = (long long)seq_len_k * num_kv_heads * D_HEAD;
    const long long stride_seq_kv = (long long)num_kv_heads * D_HEAD;
    const long long stride_head_kv = (long long)D_HEAD;

    // Offset pointers
    Q += batch_idx * stride_batch_q;
    O += batch_idx * stride_batch_q;
    K += batch_idx * stride_batch_kv;
    V += batch_idx * stride_batch_kv;

    __shared__ float q_tile[Br][D_HEAD];
    __shared__ float k_tile[Bc][D_HEAD];
    __shared__ float v_tile[Bc][D_HEAD]; // Pass 2 用

    constexpr int ROWS_PER_THREAD = (Br + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 准备工作：加载 Q Tile
    for (int i = thread_id; i < Br * D_HEAD; i += BLOCK_SIZE) {
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        int q_global_row = q_block_start_row + row;
        if (row < Br && q_global_row < seq_len_q) {
            q_tile[row][col] = Q[q_global_row * stride_seq_q + q_head_idx * stride_head_q + col];
        } else {
            q_tile[row][col] = 0.0f;
        }
    }
    __syncthreads();

    // 线程私有变量：存储当前 Q 行的全局统计信息
    float m_global[ROWS_PER_THREAD];     // 全局最大值
    float l_global[ROWS_PER_THREAD];     // 全局分母
    float acc[ROWS_PER_THREAD][D_HEAD];  // 分子累加器

    // 初始化
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        m_global[i] = -INFINITY;
        l_global[i] = 0.0f;
        for (int d = 0; d < D_HEAD; ++d) acc[i][d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D_HEAD);

    // Pass 1: 遍历 K，计算 Global Max (m) 和 Sum Exp (l)
    for (int col_block_start = 0; col_block_start < seq_len_k; col_block_start += Bc) {
        // Load K Tile
        for (int i = thread_id; i < Bc * D_HEAD; i += BLOCK_SIZE) {
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            int k_global_row = col_block_start + row;
            if (row < Bc && k_global_row < seq_len_k)
                k_tile[row][col] = K[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
            else
                k_tile[row][col] = 0.0f;
        }
        __syncthreads();

        // Compute QK^T and Update Max
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

            for (int k = 0; k < Bc; ++k) {
                int k_global_col = col_block_start + k;
                if (k_global_col >= seq_len_k) continue;
                if (is_causal && k_global_col > q_global_row) continue;

                float score = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) {
                    score += q_tile[q_local_row][d] * k_tile[k][d];
                }
                score *= scale;
                m_global[i] = fmaxf(m_global[i], score);
            }
        }
        __syncthreads();
    }
    
    // 原论文里面的online方法精度过不去，这里改成苯办法，offline去做
    for (int col_block_start = 0; col_block_start < seq_len_k; col_block_start += Bc) {
         // Reload K
        for (int i = thread_id; i < Bc * D_HEAD; i += BLOCK_SIZE) {
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            int k_global_row = col_block_start + row;
            if (row < Bc && k_global_row < seq_len_k)
                k_tile[row][col] = K[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
            else
                k_tile[row][col] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

            for (int k = 0; k < Bc; ++k) {
                int k_global_col = col_block_start + k;
                if (k_global_col >= seq_len_k) continue;
                if (is_causal && k_global_col > q_global_row) continue;

                float score = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) score += q_tile[q_local_row][d] * k_tile[k][d];
                score *= scale;
                l_global[i] += expf(score - m_global[i]); 
            }
        }
        __syncthreads();
    }

    // Pass 2: 再次遍历 K/V，计算分子 Accumulate(P * V)
    for (int col_block_start = 0; col_block_start < seq_len_k; col_block_start += Bc) {
        // Load K and V
        for (int i = thread_id; i < Bc * D_HEAD; i += BLOCK_SIZE) {
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            int k_global_row = col_block_start + row;
            if (row < Bc && k_global_row < seq_len_k) {
                k_tile[row][col] = K[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
                v_tile[row][col] = V[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
            } else {
                k_tile[row][col] = 0.0f; 
                v_tile[row][col] = 0.0f;
            }
        }
        __syncthreads();

        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

            for (int k = 0; k < Bc; ++k) {
                int k_global_col = col_block_start + k;
                if (k_global_col >= seq_len_k) continue;
                if (is_causal && k_global_col > q_global_row) continue;

                float score = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) score += q_tile[q_local_row][d] * k_tile[k][d];
                score *= scale;

                // 计算 P_ij (无需任何 Rescaling，直接算)
                float p_val = expf(score - m_global[i]); 
                
                // 累加 P * V
                for (int d = 0; d < D_HEAD; ++d) {
                    acc[i][d] += p_val * v_tile[k][d];
                }
            }
        }
        __syncthreads();
    }

    // Final: Write Output
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int q_local_row = thread_id * ROWS_PER_THREAD + i;
        int q_global_row = q_block_start_row + q_local_row;
        if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

        float inv_l = (l_global[i] == 0.0f) ? 0.0f : (1.0f / l_global[i]);
        for (int d = 0; d < D_HEAD; ++d) {
            O[q_global_row * stride_seq_q + q_head_idx * stride_head_q + d] = acc[i][d] * inv_l;
        }
    }
}
template <int Br, int Bc, int D_HEAD, int BLOCK_SIZE>
__global__ void flash_fwd_kernel_half(const half *Q, const half *K, const half *V, half *O, int batch_size,
                                      int num_q_heads, int num_kv_heads, int seq_len_q, int seq_len_k, bool is_causal) {
    int batch_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int row_block_idx = blockIdx.z;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / gqa_ratio;

    const int q_block_start_row = row_block_idx * Br;
    const int thread_id = threadIdx.x;

    const long long stride_batch_q = (long long)seq_len_q * num_q_heads * D_HEAD;
    const long long stride_seq_q = (long long)num_q_heads * D_HEAD;
    const long long stride_head_q = (long long)D_HEAD;

    const long long stride_batch_kv = (long long)seq_len_k * num_kv_heads * D_HEAD;
    const long long stride_seq_kv = (long long)num_kv_heads * D_HEAD;
    const long long stride_head_kv = (long long)D_HEAD;

    Q += batch_idx * stride_batch_q;
    O += batch_idx * stride_batch_q;
    K += batch_idx * stride_batch_kv;
    V += batch_idx * stride_batch_kv;

    __shared__ half q_tile_fp16[Br][D_HEAD];
    __shared__ half k_tile_fp16[Bc][D_HEAD];
    __shared__ half v_tile_fp16[Bc][D_HEAD];

    constexpr int ROWS_PER_THREAD = (Br + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Fix 1: Use float for accumulators and statistics to avoid overflow/precision loss
    float o_accumulator[ROWS_PER_THREAD][D_HEAD];
    float m_i[ROWS_PER_THREAD];
    float l_i[ROWS_PER_THREAD];

    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
        for (int d = 0; d < D_HEAD; ++d)
            o_accumulator[i][d] = 0.0f;
    }

    // Load Q
    for (int i = thread_id; i < Br * D_HEAD; i += BLOCK_SIZE) {
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        int q_global_row = q_block_start_row + row;
        if (row < Br && q_global_row < seq_len_q) {
            q_tile_fp16[row][col] = Q[q_global_row * stride_seq_q + q_head_idx * stride_head_q + col];
        } else {
            q_tile_fp16[row][col] = 0.0f;
        }
    }
    __syncthreads();

    for (int col_block_start = 0; col_block_start < seq_len_k; col_block_start += Bc) {
        // Load K/V
        for (int i = thread_id; i < Bc * D_HEAD; i += BLOCK_SIZE) {
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            int k_global_row = col_block_start + row;
            if (row < Bc && k_global_row < seq_len_k) {
                k_tile_fp16[row][col] = K[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
                v_tile_fp16[row][col] = V[k_global_row * stride_seq_kv + kv_head_idx * stride_head_kv + col];
            } else {
                k_tile_fp16[row][col] = 0.0f;
                v_tile_fp16[row][col] = 0.0f;
            }
        }
        __syncthreads();

        // 很难绷的是，half的精度过得去。所以这里用论文里面的one pass方案。

        float s_scores[ROWS_PER_THREAD][Bc];
        const float scale = 1.0f / sqrtf((float)D_HEAD);

        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q)
                continue;

            for (int k_local_col = 0; k_local_col < Bc; ++k_local_col) {
                float sum = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) {
                    sum += (float)q_tile_fp16[q_local_row][d] * (float)k_tile_fp16[k_local_col][d];
                }
                s_scores[i][k_local_col] = sum * scale;

                int k_global_col = col_block_start + k_local_col;
                if (k_global_col >= seq_len_k || (is_causal && k_global_col > q_global_row)) {
                    s_scores[i][k_local_col] = -INFINITY;
                }
            }
        }

        // Softmax Update
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q)
                continue;

            float m_block = -INFINITY;
            for (int k = 0; k < Bc; ++k)
                m_block = fmaxf(m_block, s_scores[i][k]);
            if (m_block == -INFINITY) continue;

            float m_old = m_i[i];
            float m_new = fmaxf(m_old, m_block);

            float l_block = 0.0f;
            for (int k = 0; k < Bc; ++k) {
                s_scores[i][k] = expf(s_scores[i][k] - m_new);
                l_block += s_scores[i][k];
            }
            float l_old = l_i[i];
            float l_new = l_old * expf(m_old - m_new) + l_block;

            float scale_o = (l_new == 0.0f) ? 0.0f : (l_old * expf(m_old - m_new) / l_new);
            
            for (int d = 0; d < D_HEAD; ++d)
                o_accumulator[i][d] *= scale_o;

            for (int k = 0; k < Bc; ++k) {
                float p_val = s_scores[i][k];
                float factor = (l_new == 0.0f) ? 0.0f : (p_val / l_new);
                for (int d = 0; d < D_HEAD; ++d) {
                    // Cast V to float for accumulation
                    o_accumulator[i][d] += factor * (float)v_tile_fp16[k][d];
                }
            }

            m_i[i] = m_new;
            l_i[i] = l_new;
        }
        __syncthreads();
    }

    // Write result
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int q_local_row = thread_id * ROWS_PER_THREAD + i;
        int q_global_row = q_block_start_row + q_local_row;
        if (q_local_row >= Br || q_global_row >= seq_len_q)
            continue;
        for (int d = 0; d < D_HEAD; ++d) {
            // Fix 5: Cast float accumulator back to half for output
            O[q_global_row * stride_seq_q + q_head_idx * stride_head_q + d] = (half)o_accumulator[i][d];
        }
    }
}

// --- Kernel 分发器 ---
// 这个函数根据运行时的 head_dim 选择一个编译好的 kernel 版本
template <typename T>
void launch_flash_fwd_kernel(const T *d_q, const T *d_k, const T *d_v, T *d_o, int batch_size, int target_seq_len,
                             int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal) {
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 32;

    dim3 grid(batch_size, query_heads, (target_seq_len + Br - 1) / Br);
    dim3 block(BLOCK_SIZE);
    // throw std::runtime_error("Unsupported head_dim in flashAttention. "
    //                            "Please compile a kernel version for head_dim=" + std::to_string(head_dim) +
    //                         "current function call:" + std::to_string(batch_size) + "," +
    //                         std::to_string(target_seq_len) + ","
    //                         + std::to_string(src_seq_len) + "," + std::to_string(query_heads) + ","
    //                         + std::to_string(kv_heads) + "," + std::to_string(head_dim) + "," +
    //                         std::to_string(is_causal));
    
    if constexpr (std::is_same<T, float>::value) {
        if (head_dim == 1) {
            flash_fwd_kernel_float<Br, Bc, 1, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 2) {
            flash_fwd_kernel_float<Br, Bc, 2, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 4) {
            flash_fwd_kernel_float<Br, Bc, 4, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 8) {
            flash_fwd_kernel_float<Br, Bc, 8, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 16) {
            flash_fwd_kernel_float<Br, Bc, 16, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 24) {
            flash_fwd_kernel_float<Br, Bc, 24, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 32) {
            flash_fwd_kernel_float<Br, Bc, 32, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 64) {
            flash_fwd_kernel_float<Br, Bc, 64, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 128) {
            flash_fwd_kernel_float<Br, Bc, 128, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else {
            throw std::runtime_error("Unsupported head_dim in flashAttention. "
                                     "Please compile a kernel version for head_dim=" +
                                     std::to_string(head_dim));
        }
    } else if constexpr (std::is_same<T, half>::value) {
                if (head_dim == 1) {
            flash_fwd_kernel_half<Br, Bc, 1, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 2) {
            flash_fwd_kernel_half<Br, Bc, 2, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 4) {
            flash_fwd_kernel_half<Br, Bc, 4, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 8) {
            flash_fwd_kernel_half<Br, Bc, 8, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 16) {
            flash_fwd_kernel_half<Br, Bc, 16, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 24) {
            flash_fwd_kernel_half<Br, Bc, 24, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 32) {
            flash_fwd_kernel_half<Br, Bc, 32, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 64) {
            flash_fwd_kernel_half<Br, Bc, 64, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else if (head_dim == 128) {
            flash_fwd_kernel_half<Br, Bc, 128, BLOCK_SIZE><<<grid, block>>>(
                d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads, target_seq_len, src_seq_len, is_causal);
        } else {
            throw std::runtime_error("Unsupported head_dim in flashAttention. "
                                     "Please compile a kernel version for head_dim=" +
                                     std::to_string(head_dim));
        }

    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k, const std::vector<T> &h_v,
                    std::vector<T> &h_o, int batch_size, int target_seq_len, int src_seq_len, int query_heads,
                    int kv_heads, int head_dim, bool is_causal) {

    if (query_heads % kv_heads != 0) {
        throw std::invalid_argument("query_heads must be divisible by kv_heads for GQA.");
    }

    long long q_elements = (long long)batch_size * target_seq_len * query_heads * head_dim;
    long long k_elements = (long long)batch_size * src_seq_len * kv_heads * head_dim;
    long long v_elements = (long long)batch_size * src_seq_len * kv_heads * head_dim;
    long long o_elements = q_elements;

    if (h_q.size() != q_elements || h_k.size() != k_elements || h_v.size() != v_elements) {
        throw std::invalid_argument("Input vector sizes do not match the provided dimensions.");
    }
    if (h_o.size() != o_elements) {
        h_o.resize(o_elements);
    }
    
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_bytes = q_elements * sizeof(T);
    size_t k_bytes = k_elements * sizeof(T);
    size_t v_bytes = v_elements * sizeof(T);
    size_t o_bytes = o_elements * sizeof(T);

    RUNTIME_CHECK(cudaMalloc(&d_q, q_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_bytes));

    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice));

    launch_flash_fwd_kernel(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
                             query_heads, kv_heads, head_dim, is_causal);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));

}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &, const std::vector<half> &,
                                   std::vector<half> &, int, int, int, int, int, int, bool);

