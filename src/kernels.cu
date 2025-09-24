#include "../tester/utils.h"
#include <cstddef>
#include <iostream>
#include <vector>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

constexpr int BLOCK_SIZE = 64;

__global__ void sumKernel(int* d_in, int* d_out, size_t n) {
    __shared__ int shared_data[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < n) {
        shared_data[tid] = d_in[gid];
    } else {
        shared_data[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_out, shared_data[0]);
    }
}

__global__ void block_scan_kernel(int* d_out, const int* d_in, int* d_block_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        temp[tid] = d_in[gid];
    } else {
        temp[tid] = 0; 
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    if (tid == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = temp[tid];
        temp[tid] = 0; 
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            int swap_val = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += swap_val;
        }
        __syncthreads();
    }

    if (gid < n) {
        d_out[gid] = temp[tid];
    }
}

__global__ void add_block_sums_kernel(int* d_data_out, const int* d_scanned_block_sums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        d_data_out[gid] += d_scanned_block_sums[blockIdx.x];
    }
}


void prefix_sum(int* d_out, const int* d_in, int n) {
    if (n <= 0) return;
    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = BLOCK_SIZE * sizeof(int);

    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));

    block_scan_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(d_out, d_in, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks > 1) {
        int* d_scanned_block_sums;
        CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(int)));
        prefix_sum(d_scanned_block_sums, d_block_sums, num_blocks);
        
        add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_scanned_block_sums, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_scanned_block_sums));
    }
    
    CUDA_CHECK(cudaFree(d_block_sums));
}



template <typename T> 
__global__ void findKthLargestKernel_count(const T *d_input, T pivot, int* d_count_greater, int* d_count_less, int* d_count_equal, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        T val = d_input[gid];
        if (val > pivot) {
            d_count_greater[gid] = 1;
            d_count_less[gid] = 0;
            d_count_equal[gid] = 0;
        } else if (val < pivot) {
            d_count_greater[gid] = 0;
            d_count_less[gid] = 1;
            d_count_equal[gid] = 0;
        } else {
            d_count_greater[gid] = 0;
            d_count_less[gid] = 0;
            d_count_equal[gid] = 1;
        }
    }
}


template <typename T> 
__global__ void moveData(T *d_output, const T* d_input, const int* d_sum, const int* d_count, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && d_count[gid] == 1) {
        d_output[d_sum[gid]] = d_input[gid];
    }
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 *
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed.
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */

template <typename T> T kthLargest(const std::vector<T> &h_input, size_t k) {
    if (k < 1 || k > h_input.size()) {
        return T(-100);
    }
    
    T *d_src, *d_dst;
    int* d_count[3];
    int* d_sum[3];

    size_t n = h_input.size();

    CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_src, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMalloc(&d_count[i], n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sum[i], n * sizeof(int)));
    }

    size_t current_size = n;
    T pivot;

    while (current_size > 1) {
        int pivot_idx_in_range = rand() % current_size;
        CUDA_CHECK(cudaMemcpy(&pivot, d_src + pivot_idx_in_range, sizeof(T), cudaMemcpyDeviceToHost));
        
        int num_blocks = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        findKthLargestKernel_count<<<num_blocks, BLOCK_SIZE>>>(d_src, pivot, d_count[0], d_count[1], d_count[2], current_size);
        
        for (int i = 0; i < 3; ++i) {
            prefix_sum(d_sum[i], d_count[i], current_size);
        }
        
        // 在所有异步任务分派后，在此处同步一次，以确保 prefix_sum 全部完成
        CUDA_CHECK(cudaDeviceSynchronize());

        int greater_count = 0, less_count = 0;
        if (current_size > 0) {
            int last_val_greater, last_val_less;
            CUDA_CHECK(cudaMemcpy(&greater_count, d_sum[0] + current_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_val_greater, d_count[0] + current_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            greater_count += last_val_greater;

            CUDA_CHECK(cudaMemcpy(&less_count, d_sum[1] + current_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_val_less, d_count[1] + current_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            less_count += last_val_less;
        }
        int equal_count = current_size - greater_count - less_count;

        if (k <= greater_count) {
            moveData<<<num_blocks, BLOCK_SIZE>>>(d_dst, d_src, d_sum[0], d_count[0], current_size);
            current_size = greater_count;
        } else if (k > greater_count + equal_count) {
            moveData<<<num_blocks, BLOCK_SIZE>>>(d_dst, d_src, d_sum[1], d_count[1], current_size);
            k -= (greater_count + equal_count);
            current_size = less_count;
        } else {
            goto cleanup;
        }

        std::swap(d_src, d_dst);

        if (current_size == 0) break;
    }

    if (current_size == 1) {
        CUDA_CHECK(cudaMemcpy(&pivot, d_src, sizeof(T), cudaMemcpyDeviceToHost));
    }

cleanup:
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaFree(d_count[i]));
        CUDA_CHECK(cudaFree(d_sum[i]));
    }
    
    return pivot;
}


template<int Br, int Bc, int D_HEAD, int BLOCK_SIZE>
__global__ void flash_fwd_kernel(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* O,
    int batch_size, 
    int num_q_heads,
    int num_kv_heads,
    int seq_len_q, 
    int seq_len_k,
    bool is_causal
) {
    // 分块索引
    int batch_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int row_block_idx = blockIdx.z;

    // GQA头映射
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / gqa_ratio;
    
    // Q块起始行
    const int q_block_start_row = row_block_idx * Br;
    const int thread_id = threadIdx.x;

    // Q的数据布局: [batch, tgt_seq, q_head, head_dim]
    const long long stride_batch_q = (long long)seq_len_q * num_q_heads * D_HEAD;
    const long long stride_seq_q   = (long long)num_q_heads * D_HEAD;
    const long long stride_head_q  = (long long)D_HEAD;

    // K/V的数据布局: [batch, src_seq, kv_head, head_dim]
    const long long stride_batch_kv = (long long)seq_len_k * num_kv_heads * D_HEAD;
    const long long stride_seq_kv   = (long long)num_kv_heads * D_HEAD;
    const long long stride_head_kv  = (long long)D_HEAD;

    // 指针偏移
    Q += batch_idx * stride_batch_q;
    O += batch_idx * stride_batch_q;
    K += batch_idx * stride_batch_kv;
    V += batch_idx * stride_batch_kv;

    __shared__ float q_tile[Br][D_HEAD];
    __shared__ float k_tile[Bc][D_HEAD];
    __shared__ float v_tile[Bc][D_HEAD];

    constexpr int ROWS_PER_THREAD = (Br + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float o_accumulator[ROWS_PER_THREAD][D_HEAD];
    float m_i[ROWS_PER_THREAD];
    float l_i[ROWS_PER_THREAD];

    for(int i = 0; i < ROWS_PER_THREAD; ++i) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
        for (int d = 0; d < D_HEAD; ++d)
            o_accumulator[i][d] = 0.0f;
    }

    // 加载Q到共享内存
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

    for (int col_block_start = 0; col_block_start < seq_len_k; col_block_start += Bc) {
        // 加载K/V到共享内存
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

        // 块内计算 S_ij
        float s_scores[ROWS_PER_THREAD][Bc];
        const float scale = 1.0f / sqrtf((float)D_HEAD);

        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

            for (int k_local_col = 0; k_local_col < Bc; ++k_local_col) {
                float sum = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) {
                    sum += q_tile[q_local_row][d] * k_tile[k_local_col][d];
                }
                s_scores[i][k_local_col] = sum * scale;

                // Causal Mask
                int k_global_col = col_block_start + k_local_col;
                if (k_global_col >= seq_len_k) {
                    s_scores[i][k_local_col] = -INFINITY;
                }
                if (is_causal && k_global_col > q_global_row) {
                    s_scores[i][k_local_col] = -INFINITY;
                }
            }
        }

        // 在线softmax分块累积
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            int q_local_row = thread_id * ROWS_PER_THREAD + i;
            int q_global_row = q_block_start_row + q_local_row;
            if (q_local_row >= Br || q_global_row >= seq_len_q) continue;

            float m_block = -INFINITY;
            for (int k = 0; k < Bc; ++k)
                m_block = fmaxf(m_block, s_scores[i][k]);
            float m_old = m_i[i];
            float m_new = fmaxf(m_old, m_block);

            float l_block = 0.0f;
            for (int k = 0; k < Bc; ++k) {
                s_scores[i][k] = expf(s_scores[i][k] - m_new);
                l_block += s_scores[i][k];
            }
            float l_old = l_i[i];
            float l_new = l_old * expf(m_old - m_new) + l_block;

            // 累加输出
            float scale_o = (l_new == 0.0f) ? 0.0f : (l_old * expf(m_old - m_new) / l_new);
            for (int d = 0; d < D_HEAD; ++d)
                o_accumulator[i][d] *= scale_o;

            for (int k = 0; k < Bc; ++k) {
                float p_val = s_scores[i][k];
                for (int d = 0; d < D_HEAD; ++d) {
                    o_accumulator[i][d] += (l_new == 0.0f ? 0.0f : (p_val / l_new) * v_tile[k][d]);
                }
            }

            m_i[i] = m_new;
            l_i[i] = l_new;
        }
        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int q_local_row = thread_id * ROWS_PER_THREAD + i;
        int q_global_row = q_block_start_row + q_local_row;
        if (q_local_row >= Br || q_global_row >= seq_len_q) continue;
        for (int d = 0; d < D_HEAD; ++d) {
            O[q_global_row * stride_seq_q + q_head_idx * stride_head_q + d] = o_accumulator[i][d];
        }
    }
}

// --- Kernel 分发器 ---
// 这个函数根据运行时的 head_dim 选择一个编译好的 kernel 版本
template <typename T>
void launch_flash_fwd_kernel(
    const T* d_q, const T* d_k, const T* d_v, T* d_o,
    int batch_size, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal
) {
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int BLOCK_SIZE = 32;

    dim3 grid(batch_size, query_heads, (target_seq_len + Br - 1) / Br);
    dim3 block(BLOCK_SIZE);
    // throw std::runtime_error("Unsupported head_dim in flashAttention. "
    //                            "Please compile a kernel version for head_dim=" + std::to_string(head_dim) + 
    //                         "current function call:" + std::to_string(batch_size) + "," + std::to_string(target_seq_len) + ","
    //                         + std::to_string(src_seq_len) + "," + std::to_string(query_heads) + ","
    //                         + std::to_string(kv_heads) + "," + std::to_string(head_dim) + "," + std::to_string(is_causal));


    if(head_dim == 1) {
        flash_fwd_kernel<Br, Bc, 1, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if(head_dim == 2) {
        flash_fwd_kernel<Br, Bc, 2, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 4) {
        flash_fwd_kernel<Br, Bc, 4, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 8) {
        flash_fwd_kernel<Br, Bc, 8, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 16) {
        flash_fwd_kernel<Br, Bc, 16, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 24) {
        flash_fwd_kernel<Br, Bc, 24, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 32) {
        flash_fwd_kernel<Br, Bc, 32, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 64) {
        flash_fwd_kernel<Br, Bc, 64, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else if (head_dim == 128) {
        flash_fwd_kernel<Br, Bc, 128, BLOCK_SIZE><<<grid, block>>>(
            d_q, d_k, d_v, d_o, batch_size, query_heads, kv_heads,
            target_seq_len, src_seq_len, is_causal);
    } else {
        throw std::runtime_error("Unsupported head_dim in flashAttention. "
                               "Please compile a kernel version for head_dim=" + std::to_string(head_dim));
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
                    int kv_heads, int head_dim, bool is_causal)
{
    static_assert(std::is_same<T, float>::value, "This implementation currently only supports float.");

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

    CUDA_CHECK(cudaMalloc(&d_q, q_bytes));
    CUDA_CHECK(cudaMalloc(&d_k, k_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, v_bytes));
    CUDA_CHECK(cudaMalloc(&d_o, o_bytes));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice));

    launch_flash_fwd_kernel(d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
                             query_heads, kv_heads, head_dim, is_causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
}
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int> &, size_t);
template float kthLargest<float>(const std::vector<float> &, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &,
                                    std::vector<float> &, int, int, int, int, int, int, bool);
