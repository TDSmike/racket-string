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
    for (size_t i = 0; i < vocab.size() - 1; ++i) {
        std::string pair = vocab[i] + " " + vocab[i + 1];
        pairFrequencies[pair]++;
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


    int num_threads = 16;
    int segment_size = vocab.size() / num_threads;

    std::vector<std::string> final_vocab;
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int start_index = thread_id * segment_size;
        int end_index = (thread_id == num_threads - 1)? vocab.size() : (thread_id + 1) * segment_size;


        std::vector<std::string> localVocab(vocab.begin() + start_index, vocab.begin() + end_index);
        int localVocabSize = localVocab.size();


        while (localVocabSize > 10) {
            std::unordered_map<std::string, int> pairFrequencies = getPairFrequencies(localVocab);
            std::string mostFrequentPair = findMostFrequentPair(pairFrequencies);
            if (mostFrequentPair.empty()) break;
            localVocab = mergePair(localVocab, mostFrequentPair);
            localVocabSize = localVocab.size();
        }
        #pragma omp critical
        {
            // 将结果合并到全局的 vocab 中
            final_vocab.insert(final_vocab.end(), localVocab.begin(), localVocab.end());
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    std::ofstream outFile("output.txt");
    for (const std::string& token : final_vocab) {
        outFile << token << std::endl;
    }


    return 0;
}