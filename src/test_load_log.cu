#include "MyStringColumn.cuh"
#include "StringColumnLoader.cuh"
#include "NFALoader.h"
#include <iostream>


int CPU_sum_up_first_char(MyStringColumn& stringcolumn) {
    int result = 0;
    assert(!stringcolumn.onDevice);
    for (int idx = 0; idx < stringcolumn.metadata.num_rows; idx++) {
        if (stringcolumn.metadata.lengths[idx] > 0) {
            result += stringcolumn.get_char(idx, 0);
        }
    }
    return result;
}


int main(int argc, char** argv) {

    printf("Testing if things work");
    cudaStream_t our_stream;
    cuda_err_chk(cudaStreamCreate(&our_stream));
    std::chrono::high_resolution_clock::time_point kernel_start;
    std::chrono::high_resolution_clock::time_point kernel_end;

    kernel_start = std::chrono::high_resolution_clock::now();

    std::string automata_filename = "regexNFA.anml";
    auto column = getStringColumnFromTextFile("Windows100000.log");
    column.transpose();
    MyStringColumn* column_d = MyStringColumn::CopyToDevice(column, false);
    MyStringColumn column_d_to_h = MyStringColumn::CopyToHost(column_d);
    assert(column_d_to_h.data_size == column.data_size);


    MyStringColumn* column_d_transposed = MyStringColumn::CopyToDevice(column, true);
    
    // ADD CODE TO LOAD NFA
    // auto nfa = load_nfa_from_file(automata_filename);

    column.free_();
    MyStringColumn::FreeStructOnDevice_(column_d);



    kernel_end = std::chrono::high_resolution_clock::now();

    cuda_err_chk(cudaStreamSynchronize(our_stream));
    std::cout << "Kernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count() << " microseconds" << std::endl;
    return 0;
}