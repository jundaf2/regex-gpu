#include "utils.h"
#include "Regex.cuh"
#include <iostream>
#include "StringColumnLoader.cuh"
#include "MySimpleNFA.h"
#include <string>
#include "vasim_helper.h"
#include "helper_string.h"



void printHelp(void) {
    printf("Usage: test_loadNFA.cu --nfafile= --logfile= \n");
    printf("file needs to be the ABSOLUTE PATH TO THE FILE\n");
    printf("example: ./build/exe/src/main.cu.exe --logfile=/home/regex-datasets/Windows_double_quote_repaired_string_column.log --nfafile=/home/kwu/ece508project/datasets/log_query1.libfsm_output --output=show\n");
}

int main(const int argc, const char **argv){
    char* nfafile = nullptr; char* logfile = nullptr; char* result = nullptr;
    if (!checkCmdLineFlag(argc, argv, "logfile") || !checkCmdLineFlag(argc, argv, "nfafile")) {
        printf("Error: mode or flag option not found\n");
        printHelp();
        return 1;
    }
    getCmdLineArgumentString(argc, argv, "nfafile", &nfafile);
    getCmdLineArgumentString(argc, argv, "logfile", &logfile);

    std::string show_result;
    if (checkCmdLineFlag(argc, argv, "output")) {
        getCmdLineArgumentString(argc, argv, "output", &result);
        show_result = result;
    }


    printf("Testing if things work\n");
    cudaStream_t our_stream;
    cuda_err_chk(cudaStreamCreate(&our_stream));
    std::chrono::high_resolution_clock::time_point timer_start;
    std::chrono::high_resolution_clock::time_point timer_end;

    timer_start = std::chrono::high_resolution_clock::now();

    std::string nfa_filename = nfafile;
    std::string log_filename = logfile;
    
    // load log
    MyStringColumn column = getStringColumnFromTextFile(log_filename.c_str());
    timer_end = std::chrono::high_resolution_clock::now();
    std::cout << "Load LOG: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    //column.transpose();
    MyStringColumn* column_d = MyStringColumn::CopyToDevice(column, false);

    // ADD CODE TO LOAD NFA
    timer_start = std::chrono::high_resolution_clock::now();
    MySimpleNFA test_nfa = LoadNFAFromLibFSMOutput(nfa_filename.c_str());
    timer_end = std::chrono::high_resolution_clock::now();
    std::cout << "Load NFA: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    //test_nfa.printTransitions();
    int* row_match_result = new int[column.metadata.num_rows];
    int* row_match_result_d;
    int* row_match_result_h;
    cuda_err_chk(cudaMalloc((void**)&row_match_result_d, (column.metadata.num_rows) * sizeof(int)));
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    row_match_result_h = (int*)malloc((column.metadata.num_rows) * sizeof(int));

    MyGPUNFA nfa_gpu(test_nfa);
    MyGPUNFA* nfa_gpu_d = MyGPUNFA::CopyToDevice(nfa_gpu);
    
    int count;
    
    // GPU version: share work load in a block with both nfa and string data shared
    std::cout << "BQ_regex_gpu3 shared memory usage: " << (7 * 512 * 4 + nfa_gpu.total_num_of_bytes) << " bytes" << std::endl;
    BQ_regex_gpu3 << < column.metadata.num_rows, NUM_THREADS, nfa_gpu.total_num_of_bytes >> > (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "BQ_regex_gpu3: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;
    count = 0;
    if(show_result=="show"){
        for (int i = 0; i < column.metadata.num_rows && count<20; i++) {
            if (row_match_result_h[i] != 0) {
                count++;
                std::cout << "Block GPU 3 Found at row " << i << std::endl;
                row_match_result_h[i] = 0; // back to zero
            }
        }
    }
    
    
    //// GPU version: BQ with less blocks than row number
    //timer_start = std::chrono::high_resolution_clock::now(); 
    //BQ_regex_gpu2 <<<1000, 512>> > (column_d, nfa_gpu_d, row_match_result_d);
    //cuda_err_chk(cudaDeviceSynchronize());
    //timer_end = std::chrono::high_resolution_clock::now();
    //cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "BQ_regex_gpu2: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;
    //for (int i = 0; i < column.metadata.num_rows; i++) {
    //    if (row_match_result_h[i] != 0) {
    //        std::cout << "Block GPU 2 Found at row " << i << std::endl;
    //        row_match_result_h[i] = 0; // back to zero
    //    }
    //}

    //// GPU version: share work load in a block with string data shared
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    timer_start = std::chrono::high_resolution_clock::now();
    BQ_regex_gpu1 << <column.metadata.num_rows, NUM_THREADS >> > (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "BQ_regex_gpu1: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;
    
    count = 0;
    if (show_result == "show")
    for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
        if (row_match_result_h[i] != 0) {
            count++;
            std::cout << "Block GPU 1 Found at row " << i << std::endl;
            row_match_result_h[i] = 0; // back to zero
        }
    }

    //// GPU version: share work load in a block
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    timer_start = std::chrono::high_resolution_clock::now();
    BQ_regex_gpu << <column.metadata.num_rows, NUM_THREADS >> > (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "BQ_regex_gpu: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    count = 0;
    if (show_result == "show")
    for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
        if (row_match_result_h[i] != 0) {
            count++;
            std::cout << "Block GPU Found at row " << i << std::endl;
            row_match_result_h[i] = 0; // back to zero
        }
    }


    ////// GPU version: block number < num of entries
    //cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    //timer_start = std::chrono::high_resolution_clock::now();
    //base_regex_gpu1 <<<1024, 512>>> (column_d, nfa_gpu_d, row_match_result_d);
    //cuda_err_chk(cudaDeviceSynchronize());
    //timer_end = std::chrono::high_resolution_clock::now();
    //cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "base_regex_gpu1: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;
    //for (int i = 0; i < column.metadata.num_rows; i++) {
    //    if (row_match_result_h[i] != 0) {
    //        std::cout << "GPU1 Found at row " << i << std::endl;
    //        row_match_result_h[i] = 0; // back to zero
    //    }
    //}

    // GPU version: block number = num of entries
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    timer_start = std::chrono::high_resolution_clock::now();
    base_regex_gpu << <column.metadata.num_rows, NUM_THREADS >> > (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "base_regex_gpu: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    count = 0;
    if (show_result == "show")
    for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
        if (row_match_result_h[i] != 0) {
            count++;
            std::cout << "GPU Found at row " << i << std::endl;
            row_match_result_h[i] = 0; // back to zero
        }
    }


    // GPU: NFA shared memory
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    timer_start = std::chrono::high_resolution_clock::now();
    base_regex_gpu2 << <column.metadata.num_rows, NUM_THREADS, nfa_gpu.total_num_of_bytes >> > (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "base_regex_gpu2: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    count = 0;
    if (show_result == "show")
    for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
        if (row_match_result_h[i] != 0) {
            count++;
            std::cout << "GPU 2 Found at row " << i << std::endl;
            row_match_result_h[i] = 0; // back to zero
        }
    }

    // GPU: Block coarsening
    cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    timer_start = std::chrono::high_resolution_clock::now();
    block_coarsening_regex_gpu <<<(column.metadata.num_rows-1)/ENTRIES_BLOCK+1, NUM_THREADS, nfa_gpu.total_num_of_bytes >>> (column_d, nfa_gpu_d, row_match_result_d);
    cuda_err_chk(cudaDeviceSynchronize());
    timer_end = std::chrono::high_resolution_clock::now();
    cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "block_coarsening_regex_gpu: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    count = 0;
    if (show_result == "show")
        for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
            if (row_match_result_h[i] != 0) {
                count++;
                std::cout << "GPU Block Coarsening Found at row " << i << std::endl;
                row_match_result_h[i] = 0; // back to zero
            }
        }
    
  
    // GPU: dynamic parallelism 
    //cuda_err_chk(cudaMemset(row_match_result_d, 0, (column.metadata.num_rows) * sizeof(int)));
    //timer_start = std::chrono::high_resolution_clock::now();
    //dynamic_para_regex_gpu << <(column.metadata.num_rows-1)/512+1, 512, nfa_gpu.total_num_of_bytes >> > (column_d, nfa_gpu_d, row_match_result_d);
    //cuda_err_chk(cudaDeviceSynchronize());
    //timer_end = std::chrono::high_resolution_clock::now();
    //cuda_err_chk(cudaMemcpy(row_match_result_h, row_match_result_d, (column.metadata.num_rows) * sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "dynamic_para_regex_gpu: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;

    //count = 0;
    //if (show_result == "show")
    //    for (int i = 0; i < column.metadata.num_rows && count < 20; i++) {
    //        if (row_match_result_h[i] != 0) {
    //            count++;
    //            std::cout << "GPU dynamic parallelism Found at row " << i << std::endl;
    //            row_match_result_h[i] = 0; // back to zero
    //        }
    //    }

    // CPU version
   /* timer_start = std::chrono::high_resolution_clock::now();
    base_regex_cpu(&column, &test_nfa, row_match_result);
    timer_end = std::chrono::high_resolution_clock::now();
    std::cout << "base_regex_cpu: " << std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() << " microseconds" << std::endl;*/


    // free memory
    nfa_gpu.free_();
    free(row_match_result);
    free(row_match_result_h);
    cuda_err_chk(cudaFree(row_match_result_d));
    cuda_err_chk(cudaFree(nfa_gpu_d));
    column.free_();
    MyStringColumn::FreeStructOnDevice_(column_d);
    
    cuda_err_chk(cudaStreamSynchronize(our_stream));
    return 0;
}

