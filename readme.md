- simple_bpe is the naive implement
- all_parallel is to split corpus and parallel as a whole
- bpe_within_parallel is to parallel in functions
- cuda_bpe is to parallel by cuda


- My system is WSL2 and my result are as follows:


(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ g++ simple_bpe.cpp -o simple_bpe.out
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ ./simple_bpe.out 
Time taken: 17.2288 seconds
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ g++ all_parallel.cpp -fopenmp -o all_parallel.out
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ ./all_parallel.out 
Time taken: 0.565909 seconds
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ g++ bpe_within_parallel.cpp -fopenmp -o within_parallel.out
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ ./within_parallel.out 
Time taken: 9.21252 seconds
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ nvcc cuda_bpe.cu -o cuda_bpe.out
(base) mike@SK-WQPPOXABSSDJ:~/cuda_learning$ ./cuda_bpe.out 
Time taken: 20.7256 seconds