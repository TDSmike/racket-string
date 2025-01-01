#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <chrono>
#include <omp.h>

// 读取文件内容
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

// 初始化词汇表
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

// 统计相邻词对的频率
std::unordered_map<std::string, int> getPairFrequencies(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> pairFrequencies;
    // 定义每个线程处理的最小元素对数量
    const size_t minElementsPerThread = 10; 
    size_t numThreads = std::max(1, static_cast<int>(vocab.size() / minElementsPerThread));
    numThreads = std::min(numThreads, static_cast<size_t>(omp_get_max_threads()));

    std::vector<std::unordered_map<std::string, int>> threadResults(numThreads);

    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        size_t start = (vocab.size() * tid) / numThreads;
        size_t end = (vocab.size() * (tid + 1)) / numThreads;
        #pragma omp for nowait
        for (size_t i = start; i < end - 1; ++i) {
            std::string pair = vocab[i] + " " + vocab[i + 1];
            // 使用原子操作增加元素对的计数，避免数据竞争
            #pragma omp atomic
            threadResults[tid][pair]++;
        }
    }

    for (const auto& result : threadResults) {
        for (const auto& [pair, count] : result) {
            pairFrequencies[pair] += count;
        }
    }

    return pairFrequencies;
}

// 找到频率最高的词对
std::string findMostFrequentPair(const std::unordered_map<std::string, int>& pairFrequencies) {
    int maxFreq = 0;
    std::string mostFrequentPair;
    for (const auto& pair : pairFrequencies) {
        if (pair.second > maxFreq) {
            maxFreq = pair.second;
            mostFrequentPair = pair.first;
        }
    }
    return mostFrequentPair;
}

// 合并词汇表中的词对并去除重复
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
        vocab = mergePair(vocab, mostFrequentPair);
        vocabSize = vocab.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    std::ofstream outFile("inside_parallel_output.txt");
    for (const std::string& token : vocab) {
        outFile << token << std::endl;
    }
    return 0;
}