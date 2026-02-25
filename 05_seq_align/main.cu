#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

#define MATCH_SCORE 2
#define MISMATCH_SCORE -1
#define GAP_PENALTY -1
#define MAX_READ_LEN 256
#define KMER_SIZE 16

struct Read {
    std::string name;
    std::string seq;
};

void load_reference(const std::string& filename, std::vector<std::string>& names, std::string& concat_seq, std::vector<size_t>& offsets) {
    std::ifstream file(filename);
    if (!file.is_open()) exit(1);
    std::string line, name, seq;
    bool is_fastq = false;

    if (std::getline(file, line)) {
        if (line[0] == '@') is_fastq = true;
        name = line.substr(1);
    }

    while (std::getline(file, line)) {
        if (is_fastq) {
            seq = line;
            std::getline(file, line); 
            std::getline(file, line); 
            names.push_back(name);
            offsets.push_back(concat_seq.size());
            concat_seq += seq;
            if (std::getline(file, line)) name = line.substr(1);
        } else {
            if (line[0] == '>') {
                names.push_back(name);
                offsets.push_back(concat_seq.size());
                concat_seq += seq;
                name = line.substr(1);
                seq = "";
            } else {
                seq += line;
            }
        }
    }
    if (!is_fastq && !name.empty()) {
        names.push_back(name);
        offsets.push_back(concat_seq.size());
        concat_seq += seq;
    }
}

void load_reads(const std::string& filename, std::vector<Read>& reads) {
    std::ifstream file(filename);
    if (!file.is_open()) exit(1);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        Read r;
        r.name = line.substr(1);
        std::getline(file, r.seq);
        std::getline(file, line); 
        std::getline(file, line); 
        reads.push_back(r);
    }
}

inline uint8_t char2val_cpu(char c) {
    return (c >> 1) & 0x03;
}

inline uint32_t pack_16mer(const char* seq) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i) {
        uint8_t val = char2val_cpu(seq[i]);
        packed |= (val << (30 - 2 * i));
    }
    return packed;
}

void encode_sequence_cpu(const std::string& seq, uint32_t* buffer) {
    size_t length = seq.length();
    size_t encoded_size = (length + 15) / 16;
    
    #pragma omp parallel for
    for (size_t i = 0; i < encoded_size; ++i) {
        uint32_t packed = 0;
        size_t start = i * 16;
        for (int j = 0; j < 16; ++j) {
            if (start + j < length) {
                uint8_t val = char2val_cpu(seq[start + j]);
                packed |= (val << (30 - 2 * j));
            }
        }
        buffer[i] = packed;
    }
}

__device__ inline uint8_t get_base(const uint32_t* packed_array, size_t index) {
    size_t array_idx = index / 16;
    int bit_pos = 30 - 2 * (index % 16);
    return (packed_array[array_idx] >> bit_pos) & 0x03;
}

__global__ void sw_extend_batched_kernel(
    const uint32_t* ref_packed, size_t ref_len,
    const uint32_t* reads_packed, const int* read_lengths,
    const int* candidate_positions, const int* candidate_read_indices,
    int total_candidates,
    int* out_scores, int* out_best_pos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_candidates) return;

    int ref_start = candidate_positions[tid];
    int read_idx = candidate_read_indices[tid];
    int read_len = read_lengths[read_idx];
    
    const uint32_t* my_read_packed = reads_packed + read_idx * (MAX_READ_LEN / 16);

    // Give a larger window (100 bp extra) to accommodate high indel rates
    int ref_region_len = read_len + 100; 
    if (ref_start + ref_region_len > ref_len) {
        ref_region_len = ref_len - ref_start;
    }

    int H_row[MAX_READ_LEN];
    int Start_row[MAX_READ_LEN];
    for (int i = 0; i < read_len; ++i) {
        H_row[i] = 0;
        Start_row[i] = -1;
    }

    int max_score = 0;
    int best_ref_start = ref_start;

    for (int i = 0; i < ref_region_len; ++i) {
        int current_ref_pos = ref_start + i;
        uint8_t ref_base = get_base(ref_packed, current_ref_pos);
        
        int h_diag = 0; 
        int start_diag = current_ref_pos; 
        int h_left = 0;
        int start_left = -1;

        for (int j = 0; j < read_len; ++j) {
            uint8_t read_base = get_base(my_read_packed, j);
            int match = (ref_base == read_base) ? MATCH_SCORE : MISMATCH_SCORE;
            
            int score_diag = h_diag + match;
            int score_up = H_row[j] + GAP_PENALTY;
            int score_left = h_left + GAP_PENALTY;
            
            int score = 0;
            int start = current_ref_pos;
            
            if (score_diag > 0 && score_diag >= score_up && score_diag >= score_left) {
                score = score_diag;
                // If previous score was 0, alignment restarts HERE, update start pos
                start = (h_diag == 0) ? current_ref_pos : start_diag; 
            } else if (score_up > 0 && score_up >= score_left) {
                score = score_up;
                start = (H_row[j] == 0) ? current_ref_pos : Start_row[j];
            } else if (score_left > 0) {
                score = score_left;
                start = (h_left == 0) ? current_ref_pos : start_left;
            }

            h_diag = H_row[j];
            start_diag = Start_row[j];
            
            H_row[j] = score;  
            Start_row[j] = start;
            
            h_left = score;    
            start_left = start;

            if (score > max_score) {
                max_score = score;
                best_ref_start = start;
            }
        }
    }

    out_scores[tid] = max_score;
    out_best_pos[tid] = best_ref_start; 
}

int main() {
    std::string ref_file = "reference.fasta";
    std::string reads_file = "reads.fastq";
    std::ofstream out_file("results.txt");

    std::vector<std::string> ref_names;
    std::vector<size_t> ref_offsets;
    std::string concat_ref;

    load_reference(ref_file, ref_names, concat_ref, ref_offsets);
    
    std::vector<Read> reads;
    load_reads(reads_file, reads);

    std::unordered_map<uint32_t, std::vector<int>> kmer_index;
    // CRITICAL FIX: Stride = 1 guarantees we index EVERY possible 16-mer in the reference
    for (size_t i = 0; i + KMER_SIZE <= concat_ref.length(); i += 1) { 
        uint32_t kmer = pack_16mer(concat_ref.c_str() + i);
        if (kmer_index[kmer].size() < 100) { 
            kmer_index[kmer].push_back(i);
        }
    }

    size_t ref_encoded_size = (concat_ref.length() + 15) / 16;
    uint32_t* h_ref_packed = new uint32_t[ref_encoded_size];
    encode_sequence_cpu(concat_ref, h_ref_packed);

    uint32_t* d_ref_packed;
    cudaMalloc(&d_ref_packed, ref_encoded_size * sizeof(uint32_t));
    cudaMemcpy(d_ref_packed, h_ref_packed, ref_encoded_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int num_reads = reads.size();
    std::vector<uint32_t> h_reads_packed(num_reads * (MAX_READ_LEN / 16), 0);
    std::vector<int> h_read_lengths(num_reads, 0);
    
    std::vector<int> h_candidate_pos;
    std::vector<int> h_candidate_read_idx;

    for (int i = 0; i < num_reads; ++i) {
        int rlen = std::min((int)reads[i].seq.length(), MAX_READ_LEN);
        h_read_lengths[i] = rlen;
        
        uint32_t* read_ptr = h_reads_packed.data() + i * (MAX_READ_LEN / 16);
        encode_sequence_cpu(reads[i].seq, read_ptr);

        std::vector<int> cands;
        // CRITICAL FIX: Stride = 2 guarantees dense seed checking against the reference
        for (int offset = 0; offset <= rlen - KMER_SIZE; offset += 2) {
            uint32_t seed = pack_16mer(reads[i].seq.c_str() + offset);
            auto it = kmer_index.find(seed);
            if (it != kmer_index.end()) {
                for (int pos : it->second) {
                    // Start looking slightly before the expected position
                    int ref_window_start = std::max(0, pos - offset - 50);
                    cands.push_back(ref_window_start);
                }
            }
        }
        
        std::sort(cands.begin(), cands.end());
        int last_added = -1000;
        for(int pos : cands) {
            // De-duplicate overlapping windows
            if (pos > last_added + 50) { 
                h_candidate_pos.push_back(pos);
                h_candidate_read_idx.push_back(i);
                last_added = pos;
            }
        }
    }

    int total_candidates = h_candidate_pos.size();
    if (total_candidates == 0) {
        for (const auto& r : reads) out_file << r.name << " unknown_origin\n";
        return 0;
    }

    uint32_t* d_reads_packed;
    int *d_read_lengths, *d_candidate_pos, *d_candidate_read_idx, *d_out_scores, *d_out_best_pos;

    cudaMalloc(&d_reads_packed, h_reads_packed.size() * sizeof(uint32_t));
    cudaMalloc(&d_read_lengths, num_reads * sizeof(int));
    cudaMalloc(&d_candidate_pos, total_candidates * sizeof(int));
    cudaMalloc(&d_candidate_read_idx, total_candidates * sizeof(int));
    cudaMalloc(&d_out_scores, total_candidates * sizeof(int));
    cudaMalloc(&d_out_best_pos, total_candidates * sizeof(int));

    cudaMemcpy(d_reads_packed, h_reads_packed.data(), h_reads_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_read_lengths, h_read_lengths.data(), num_reads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate_pos, h_candidate_pos.data(), total_candidates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate_read_idx, h_candidate_read_idx.data(), total_candidates * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (total_candidates + blockSize - 1) / blockSize;
    
    sw_extend_batched_kernel<<<numBlocks, blockSize>>>(
        d_ref_packed, concat_ref.length(),
        d_reads_packed, d_read_lengths,
        d_candidate_pos, d_candidate_read_idx,
        total_candidates,
        d_out_scores, d_out_best_pos
    );
    cudaDeviceSynchronize();

    std::vector<int> h_out_scores(total_candidates);
    std::vector<int> h_out_best_pos(total_candidates);
    cudaMemcpy(h_out_scores.data(), d_out_scores, total_candidates * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_best_pos.data(), d_out_best_pos, total_candidates * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> best_score_per_read(num_reads, -1);
    std::vector<int> best_pos_per_read(num_reads, -1);

    for (int i = 0; i < total_candidates; ++i) {
        int r_idx = h_candidate_read_idx[i];
        if (h_out_scores[i] > best_score_per_read[r_idx]) {
            best_score_per_read[r_idx] = h_out_scores[i];
            best_pos_per_read[r_idx] = h_out_best_pos[i];
        }
    }

    for (int i = 0; i < num_reads; ++i) {
        // Threshold: Need at least 50% length equivalent score (~25 perfect matches minimum)
        if (best_score_per_read[i] > h_read_lengths[i] * 0.5) { 
            int global_pos = best_pos_per_read[i];
            auto it = std::upper_bound(ref_offsets.begin(), ref_offsets.end(), global_pos);
            int ref_idx = std::distance(ref_offsets.begin(), it) - 1;
            int local_pos = global_pos - ref_offsets[ref_idx];
            
            // Output format strictly matches ground truth for clean diff
            out_file << reads[i].name << " " << ref_names[ref_idx] << " " << local_pos << "\n";
        } else {
            out_file << reads[i].name << " unknown_origin\n";
        }
    }

    cudaFree(d_ref_packed);
    cudaFree(d_reads_packed);
    cudaFree(d_read_lengths);
    cudaFree(d_candidate_pos);
    cudaFree(d_candidate_read_idx);
    cudaFree(d_out_scores);
    cudaFree(d_out_best_pos);
    delete[] h_ref_packed;

    return 0;
}