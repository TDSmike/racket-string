#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file." << std::endl;
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

std::vector<std::string> initializeVocab(const std::string& text) {
    std::vector<std::string> vocab;
    std::string word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty()) {
                vocab.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) {
        vocab.push_back(word);
    }
    return vocab;
}

std::unordered_map<std::string, int> getPairFrequencies(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> pairFrequencies;
    for (size_t i = 0; i < vocab.size() - 1; ++i) {
        std::string pair = vocab[i] + " " + vocab[i + 1];
        pairFrequencies[pair]++;
    }
    return pairFrequencies;
}


__global__ void findMaxFrequencyKernel(const thrust::pair<const char*, int>* pairs, int n, thrust::pair<const char*, int>* result) {
    extern __shared__ thrust::pair<const char*, int> sharedData[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < n) {
        sharedData[tid] = pairs[i];
    } else {
        sharedData[tid] = thrust::make_pair("", -1);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s && sharedData[tid].second < sharedData[tid + s].second) {
            sharedData[tid] = sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

std::string findMostFrequentPair(const std::unordered_map<std::string, int>& pairFrequencies) {
    std::vector<thrust::pair<const char*, int>> pairs;
    for (const auto& pair : pairFrequencies) {
        pairs.push_back(thrust::make_pair(pair.first.c_str(), pair.second));
    }

    thrust::device_vector<thrust::pair<const char*, int>> d_pairs(pairs.begin(), pairs.end());
    thrust::device_vector<thrust::pair<const char*, int>> d_result(1);

    int blockSize = 256;
    int gridSize = (pairs.size() + blockSize - 1) / blockSize;
    findMaxFrequencyKernel<<<gridSize, blockSize, blockSize * sizeof(thrust::pair<const char*, int>)>>>(
        thrust::raw_pointer_cast(d_pairs.data()), pairs.size(), thrust::raw_pointer_cast(d_result.data()));
    thrust::pair<const char*, int> result = d_result[0];

    return result.first ? std::string(result.first) : "";
}

std::vector<std::string> mergePair(std::vector<std::string>& vocab, const std::string& pairToMerge) {
    std::vector<std::string> newVocab;
    std::string mergedPair = pairToMerge.substr(0, pairToMerge.find(' ')) + pairToMerge.substr(pairToMerge.find(' ') + 1);
    std::set<std::string> seen;
    bool merging = false;
    std::string currentPair;
    for (size_t i = 0; i < vocab.size() - 1; ++i) {
        currentPair = vocab[i] + " " + vocab[i + 1];
        if (currentPair == pairToMerge) {
            if (seen.find(mergedPair) == seen.end()) {
                newVocab.push_back(mergedPair);
                seen.insert(mergedPair);
            }
            merging = true;
            i++; 
        } else {
            if (merging) {
                if (seen.find(vocab[i + 1]) == seen.end()) {
                    newVocab.push_back(vocab[i + 1]);
                    seen.insert(vocab[i + 1]);
                }
                merging = false;
            } else {
                if (seen.find(vocab[i]) == seen.end()) {
                    newVocab.push_back(vocab[i]);
                    seen.insert(vocab[i]);
                }
            }
        }
    }
    if (!merging) {
        if (seen.find(vocab.back()) == seen.end()) {
            newVocab.push_back(vocab.back());
        }
    }
    return newVocab;
}

int main() {
    std::string text = readFile("large_text.txt");
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> vocab = initializeVocab(text);
    int vocabSize = vocab.size();
    while (vocabSize > 160) {
        std::unordered_map<std::string, int> pairFrequencies = getPairFrequencies(vocab);
        std::string mostFrequentPair = findMostFrequentPair(pairFrequencies);
        if (mostFrequentPair.empty()) break;
        // std::cout<<vocab.size()<<" ";
        vocab = mergePair(vocab, mostFrequentPair);
        vocabSize = vocab.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    std::ofstream outFile("cuda_output.txt");
    for (const std::string& token : vocab) {
        outFile << token << std::endl;
    }
    return 0;
}