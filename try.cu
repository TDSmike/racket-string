#include <iostream>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>


// 简单的哈希函数，将字符串映射为整数
__device__ unsigned int hash(const std::string& str) {
    unsigned int hash = 0;
    for (char c : str) {
        hash = 31 * hash + c;
    }
    return hash;
}


// 检查 CUDA 错误
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}


// CUDA 核函数，用于计算相邻词对的频率
__global__ void getPairFrequenciesKernel(const std::string* d_vocab, int vocabSize, int* d_pairFrequencies) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vocabSize - 1) {
        std::string pair = d_vocab[tid] + " " + d_vocab[tid + 1];
        unsigned int pairHash = hash(pair);
        // 使用原子操作更新频率，避免多个线程同时更新同一对的频率时的数据竞争
        atomicAdd(&d_pairFrequencies[pairHash], 1);
    }
}


// 使用 CUDA 计算相邻词对的频率
std::unordered_map<std::string, int> getPairFrequencies(const std::vector<std::string>& vocab) {
    int vocabSize = vocab.size();
    std::string* d_vocab;
    int* d_pairFrequencies;
    std::unordered_map<std::string, int> pairFrequencies;


    // 在 GPU 上分配内存
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_vocab, vocabSize * sizeof(std::string)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pairFrequencies, vocabSize * sizeof(int)));


    // 将数据传输到 GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_vocab, vocab.data(), vocabSize * sizeof(std::string), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_pairFrequencies, 0, vocabSize * sizeof(int)));


    // 启动 CUDA 核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (vocabSize + threadsPerBlock - 1) / threadsPerBlock;
    getPairFrequenciesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vocab, vocabSize, d_pairFrequencies);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // 将结果传输回 CPU
    int* h_pairFrequencies = new int[vocabSize];
    CHECK_CUDA_ERROR(cudaMemcpy(h_pairFrequencies, d_pairFrequencies, vocabSize * sizeof(int), cudaMemcpyDeviceToHost));


    // 构建 unordered_map
    for (int i = 0; i < vocabSize - 1; ++i) {
        if (h_pairFrequencies[i] > 0) {
            std::string pair = vocab[i] + " " + vocab[i + 1];
            pairFrequencies[pair] = h_pairFrequencies[i];
        }
    }


    // 清理 GPU 内存
    CHECK_CUDA_ERROR(cudaFree(d_vocab));
    CHECK_CUDA_ERROR(cudaFree(d_pairFrequencies));


    delete[] h_pairFrequencies;


    return pairFrequencies;
}


int main() {
    std::vector<std::string> vocab = {"hello", "world", "hello", "there", "world", "hello"};
    std::unordered_map<std::string, int> pairFrequencies = getPairFrequencies(vocab);
    for (const auto& pair : pairFrequencies) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    return 0;
}