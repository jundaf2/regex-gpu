#pragma once
#include "MyStringColumn.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

MyStringColumn getStringColumnFromTextFile(std::string file_name);
__global__ void sum_up_first_char(MyStringColumn* stringColumn, int* result);

void get_offsets_from_lengths(int* offsets, int* lengths, int num_rows) {
    int offset = 0;
    for (int i = 0; i < num_rows; i++) {
        offsets[i] = offset;
        offset += lengths[i];
    }
}

void get_row_lengths(int* lengths, std::vector<std::string>& log_entries, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        lengths[i] = log_entries[i].size();
    }
}


void get_row_data_from_file(std::vector<std::string>& log_entries, std::string fname, int col = 2) {
    std::ifstream myfile(fname);

    if (!myfile.is_open()) {
        std::cout << "Unable to open "+ fname;
        system("pause");
        exit(1);
    }

    // std::vector<std::string> log_file;
    std::string temp;
    while (getline(myfile, temp))
    {
        log_entries.push_back(temp);
    }

    //for (auto it = log_file.begin(); it != log_file.end(); it++)
    //{
    //    std::istringstream is(*it);
    //    std::string s;
    //    int pam = 0;
    //    while (is >> s)
    //    {
    //        if (pam == col)  // third column of windows.log

    //        {
    //            log_entries.push_back(s);
    //            log_entries.back();
    //        }
    //        pam++;
    //    }
    //}
}


void convert_vector_to_strcol(char* data, std::vector<std::string>& log_entries, int* lengths, int* offsets, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < lengths[i]; j++) {
            data[offsets[i] + j] = log_entries[i][j];
        }
    }
}

MyStringColumn getStringColumnFromTextFile(std::string file_name) {
    std::vector<std::string> log_entries;

    int* offsets;
    int* lengths;

    // read entries from log file
    get_row_data_from_file(log_entries, file_name, 4);

    // get length and offsets from file
    int num_rows = log_entries.size();
    lengths = (int*)malloc(num_rows * sizeof(int));
    offsets = (int*)malloc(num_rows * sizeof(int));

    get_row_lengths(lengths, log_entries, num_rows);
    get_offsets_from_lengths(offsets, lengths, num_rows);

    char* string_data = (char*)malloc((lengths[num_rows - 1] + offsets[num_rows - 1]) * sizeof(char));

    // fill in the data
    convert_vector_to_strcol(string_data, log_entries, lengths, offsets, num_rows);

    MyStringColumnMetadata metadata(num_rows, offsets, lengths, false);
    MyStringColumn column(string_data, metadata, lengths[num_rows - 1] + offsets[num_rows - 1], false, false);
    return column;
}

__global__ void sum_up_first_char(MyStringColumn* stringColumn, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < stringColumn->metadata.num_rows;
        i += blockDim.x * gridDim.x) {
        if (stringColumn->metadata.lengths[i] > 0) {
            atomicAdd(result, stringColumn->get_char(i, 0));
        }
    }
}