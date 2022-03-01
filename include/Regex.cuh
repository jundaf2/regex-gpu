#pragma once
#include "MyStringColumn.cuh"
#include "MySimpleNFA.h"
#include "utils.h"
#include "node.h"
#include <string>
#include <array>
#include <vector>
#include <cooperative_groups.h>

#ifndef CUDACC_RTC
#define CUDACC_RTC
#endif 

#ifndef CUDACC
#define CUDACC
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;
using namespace std;
#define TILE_SIZE 256
#define BUFFER_SIZE 30
#define NUM_THREADS 128

__global__ void my_dummy_kernel(int *a_random_ptr) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  *((int *)a_random_ptr) += id;
  return;
}

void base_regex_cpu(MyStringColumn* stringColumn, MySimpleNFA* nfa, int* result) {
	for (int bid = 0; bid < stringColumn->metadata.num_rows; bid++) {
		// cout << "##### Start row " << bid << " #####" << endl;
		const int entry_length = stringColumn->metadata.lengths[bid];
		int start_idx = 0;
		while (start_idx < entry_length) {
			int idx = 0;
			vector<int> next_level_match_node_idxes;
			vector<int> curr_level_match_node_idxes;
			curr_level_match_node_idxes.push_back(nfa->start_state);
			while (idx < entry_length - start_idx) {
				char this_char = stringColumn->get_char(bid, start_idx + idx);
				bool match_flag = false;
				for (int curr_level_node_idx : curr_level_match_node_idxes) {
					MySimpleNode curr_level_node = nfa->nodes.at(curr_level_node_idx);
					for (auto edge_pair : curr_level_node.outEdges)
					{
						int next_level_node_idx = edge_pair.first;
						for (char match_char : edge_pair.second.match_chars)
						{
							//std::cout << "isUnconditionalTransition: " << edge_pair.second.isUnconditionalTransition << " " << std::endl;
							//std::cout << "this_char: " << this_char << " " << std::endl;
							//std::cout << "match_char: " << match_char << " " << std::endl;
							if (edge_pair.second.isUnconditionalTransition || this_char == match_char) {
								if (find(next_level_match_node_idxes.begin(), next_level_match_node_idxes.end(), next_level_node_idx) == next_level_match_node_idxes.end()) {
									next_level_match_node_idxes.push_back(next_level_node_idx);
								}
								/*if (next_level_node_idx != curr_level_node_idx) {
									cout << "match_char " << match_char << endl;
								}*/
							}
						}
					}
				}
				// std::cout << std::endl;
				if (next_level_match_node_idxes.empty()) {
					// cout << "Not found!" << endl;
					break;
				}
				else {
					idx++;
					curr_level_match_node_idxes = next_level_match_node_idxes;
					next_level_match_node_idxes.clear();
					for (int final_node_idx : curr_level_match_node_idxes) {
						for (int end_node_idx : nfa->end_states) {
							if (final_node_idx == end_node_idx) {
								result[bid] = 1;
								cout << "CPU Found at row " << bid << endl; 
								match_flag = true;
							}
						}
					}
				}
				if (match_flag) break;
			}
			start_idx++;
		} // start idx
		//cout << "Finish row " << bid << endl;
	} // bid
}


/*
@ result is of size gridDim.x, which is the number of log entries in stringColumn
@ NFA defined for GPU
@ MyStringColumn both CPU and GPU
*/
__global__ void base_regex_gpu(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;

#pragma GCC ivdep //This is supposed to be equivalent to applying __restrict__ keyword to all pointers, even those as structure members.
	const int entry_length = stringColumn->metadata.lengths[bid];
	for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
		if (i + tid < min(entry_length, TILE_SIZE))
			charTile[i + tid] = stringColumn->get_char(bid, i + tid);
	}
	__syncthreads();
	
	for (int t = 0; t < min(entry_length, TILE_SIZE); t += blockDim.x) {
		if ((t + tid) < min(entry_length, TILE_SIZE)) {
			int curr_level_match_node_num = 0;
			int curr_level_match_node_idxes[BUFFER_SIZE];
			int next_level_match_node_num = 0;
			int next_level_match_node_idxes[BUFFER_SIZE];
			const int num_chars_for_search = min(entry_length - (t + tid), TILE_SIZE - (t + tid));
			int idx = 0;
			curr_level_match_node_idxes[curr_level_match_node_num++] = nfa->start_state;

			while (idx < num_chars_for_search) {
				char this_char = charTile[(t + tid) + idx];
				for (int i = 0; i < curr_level_match_node_num; i++) { // curr_level_node_idx: curr_level_match_node_idxes
					int curr_level_node_idx = curr_level_match_node_idxes[i];
					for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++) // edge_pair : curr_level_node.outEdges
					{
						int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
						for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
						{
							char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
							if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || this_char == match_char) {
								if (next_level_match_node_num < BUFFER_SIZE)
									next_level_match_node_idxes[next_level_match_node_num++] = next_level_node_idx;
							}
						}
					}
				}

				if (next_level_match_node_num == 0) {
					break;
				}
				else {
					idx++;
					for (int i = 0; i < next_level_match_node_num; i++) {// assign next to next iteration
						curr_level_match_node_idxes[i] = next_level_match_node_idxes[i];
					}
					curr_level_match_node_num = next_level_match_node_num;
					next_level_match_node_num = 0; // back to zero
					bool match_flag = false;
					for (int final_node_idx = 0; final_node_idx < curr_level_match_node_num; final_node_idx++) {
						for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
							if (curr_level_match_node_idxes[final_node_idx] == nfa->end_states[end_node_idx]) {
								atomicExch(result + bid, 1);
								match_flag = true;
							}
						}
					}
					if (match_flag) break;
				}
			}
		}
	}
}




/*
*/
__global__ void base_regex_gpu1(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;

	for(int idx_str = bid; idx_str < stringColumn->metadata.num_rows; idx_str += (gridDim.x)){
		/*if (bid == 0 && tid == 0) {
			printf("At idx_str %d of a total of %d entries.\n", idx_str, stringColumn->metadata.num_rows);
		}*/
		const int entry_length = stringColumn->metadata.lengths[idx_str];
		for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
			if (i + tid < min(entry_length, TILE_SIZE))
				charTile[i + tid] = stringColumn->get_char(idx_str, i + tid);
		}
		__syncthreads();

		const int num_chars_for_search = min(entry_length - tid, TILE_SIZE - tid);

		int idx = 0;
		int curr_level_match_node_num = 0;
		int curr_level_match_node_idxes[BUFFER_SIZE];
		int next_level_match_node_num = 0;
		int next_level_match_node_idxes[BUFFER_SIZE];
		curr_level_match_node_idxes[curr_level_match_node_num++] = nfa->start_state;
		while (idx < num_chars_for_search) {
			char this_char = charTile[tid + idx];
			for (int i = 0; i < curr_level_match_node_num; i++) { // curr_level_node_idx: curr_level_match_node_idxes
				int curr_level_node_idx = curr_level_match_node_idxes[i];
				for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++) // edge_pair : curr_level_node.outEdges
				{
					int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || this_char == match_char) {
							if (next_level_match_node_num < BUFFER_SIZE)
								next_level_match_node_idxes[next_level_match_node_num++] = next_level_node_idx;
						}
					}
				}
			}
			if (next_level_match_node_num == 0) {
				break;
			}
			else {
				idx++;
				for (int i = 0; i < next_level_match_node_num; i++) {// assign next to next iteration
					curr_level_match_node_idxes[i] = next_level_match_node_idxes[i];
				}
				curr_level_match_node_num = next_level_match_node_num;
				next_level_match_node_num = 0; // back to zero
				bool match_flag = false;
				for (int final_node_idx = 0; final_node_idx < curr_level_match_node_num; final_node_idx++) {
					for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
						if (curr_level_match_node_idxes[final_node_idx] == nfa->end_states[end_node_idx]) {
							atomicExch(result + idx_str, tid + 1);
							match_flag = true;
						}
					}
				}
				if (match_flag) break;
			}
		}
	}
}


/*
@ result is of size gridDim.x, which is the number of log entries in stringColumn
@ NFA defined for GPU
@ MyStringColumn both CPU and GPU
*/
__global__ void base_regex_gpu2(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	extern __shared__ char nfaTile[];
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;

	const int nfa_start_state = nfa->start_state;
	const int nfa_num_nodes = nfa->num_nodes;
	const int nfa_num_edges = nfa->num_edges;
	const int nfa_num_match_chars = nfa->num_match_chars;
	const int nfa_num_end_states = nfa->num_of_end_states;
	const int entry_length = stringColumn->metadata.lengths[bid];

	int* nfa_node_outEdges_nums = (int*)&nfaTile[0];
	int* nfa_node_outEdges_offsets = (int*)&nfa_node_outEdges_nums[nfa_num_nodes];
	int* nfa_outEdges_endNode = (int*)&nfa_node_outEdges_offsets[nfa_num_nodes];
	int* nfa_outEdges_match_chars_offsets = (int*)&nfa_outEdges_endNode[nfa_num_edges];
	int* nfa_outEdges_match_chars_nums = (int*)&nfa_outEdges_match_chars_offsets[nfa_num_edges];
	int* nfa_end_states = (int*)&nfa_outEdges_match_chars_nums[nfa_num_edges];
	bool* nfa_outEdges_isUnconditionalTransition = (bool*)&nfa_end_states[nfa_num_end_states];
	char* nfa_outEdges_match_chars = (char*)&nfa_outEdges_isUnconditionalTransition[nfa_num_edges];

	if (tid < nfa_num_nodes) {
		nfa_node_outEdges_nums[tid] = nfa->node_outEdges_nums[tid];
		nfa_node_outEdges_offsets[tid] = nfa->node_outEdges_offsets[tid];
	}

	if (tid < nfa_num_edges) {
		nfa_outEdges_endNode[tid] = nfa->outEdges_endNode[tid]; //printf("### nfa_outEdges_endNode\n"); 
		nfa_outEdges_isUnconditionalTransition[tid] = nfa->outEdges_isUnconditionalTransition[tid]; //printf("### nfa_outEdges_isUnconditionalTransition\n");
		nfa_outEdges_match_chars_offsets[tid] = nfa->outEdges_match_chars_offsets[tid]; //printf("### nfa_outEdges_match_chars_offsets\n");
		nfa_outEdges_match_chars_nums[tid] = nfa->outEdges_match_chars_nums[tid]; //printf("### nfa_outEdges_match_chars_nums\n");
	}

	if (tid < nfa_num_match_chars) {
		nfa_outEdges_match_chars[tid] = nfa->outEdges_match_chars[tid]; //printf("### nfa_outEdges_match_chars\n");
	}

	if (tid < nfa_num_end_states) {
		nfa_end_states[tid] = nfa->end_states[tid];
	}

	for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
		if (i + tid < min(entry_length, TILE_SIZE))
			charTile[i + tid] = stringColumn->get_char(bid, i + tid);
	}
	__syncthreads();

#pragma GCC ivdep //This is supposed to be equivalent to applying __restrict__ keyword to all pointers, even those as structure members.
	for (int t = 0; t < min(entry_length, TILE_SIZE); t += blockDim.x) {
		if ((t + tid) < min(entry_length, TILE_SIZE)) {
			int curr_level_match_node_num = 0;
			int curr_level_match_node_idxes[BUFFER_SIZE];
			int next_level_match_node_num = 0;
			int next_level_match_node_idxes[BUFFER_SIZE];
			const int num_chars_for_search = min(entry_length - (t + tid), TILE_SIZE - (t + tid));
			int idx = 0;
			curr_level_match_node_idxes[curr_level_match_node_num++] = nfa_start_state;

			while (idx < num_chars_for_search) {
				char this_char = charTile[(t + tid) + idx];
				for (int i = 0; i < curr_level_match_node_num; i++) { // curr_level_node_idx: curr_level_match_node_idxes
					int curr_level_node_idx = curr_level_match_node_idxes[i];
					for (int j = 0; j < nfa_node_outEdges_nums[curr_level_node_idx]; j++) // edge_pair : curr_level_node.outEdges
					{
						int next_level_node_idx = nfa_outEdges_endNode[nfa_node_outEdges_offsets[curr_level_node_idx] + j];
						for (int k = 0; k < nfa_outEdges_match_chars_nums[nfa_node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
						{
							char match_char = nfa_outEdges_match_chars[nfa_outEdges_match_chars_offsets[nfa_node_outEdges_offsets[curr_level_node_idx] + j] + k];
							if (nfa_outEdges_isUnconditionalTransition[nfa_node_outEdges_offsets[curr_level_node_idx] + j] || this_char == match_char) {
								if (next_level_match_node_num < BUFFER_SIZE)
									next_level_match_node_idxes[next_level_match_node_num++] = next_level_node_idx;
							}
						}
					}
				}

				if (next_level_match_node_num == 0) {
					break;
				}
				else {
					idx++;
					for (int i = 0; i < next_level_match_node_num; i++) {// assign next to next iteration
						curr_level_match_node_idxes[i] = next_level_match_node_idxes[i];
					}
					curr_level_match_node_num = next_level_match_node_num;
					next_level_match_node_num = 0; // back to zero
					bool match_flag = false;
					for (int final_node_idx = 0; final_node_idx < curr_level_match_node_num; final_node_idx++) {
						for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
							if (curr_level_match_node_idxes[final_node_idx] == nfa_end_states[end_node_idx]) {
								atomicExch(result + bid, 1);
								match_flag = true;
							}
						}
					}
					if (match_flag) break;
				}
			}
		}
	}
}
/*
Find end state together, the whole block explores the current level together!
*/
__global__ void BQ_regex_gpu(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	/*
	In this kernel, the index is for each char, which is waiting for matching,
	in the beginning, the whole entry of chars are waiting for matching.
	After the first iteration, the number should be greatly reduced.
	*/
	__shared__ bool match_flag;
	__shared__ unsigned int curr_level_match_node_num;
	__shared__ unsigned int next_level_match_node_num;
	__shared__ int curr_level_match_node_idxes[TILE_SIZE]; // start with entry_length x nfa->start_state
	__shared__ int next_level_match_node_idxes[TILE_SIZE];
	__shared__ unsigned int curr_char_indices[TILE_SIZE];
	__shared__ unsigned int next_char_indices[TILE_SIZE];

	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;	

	const int entry_length = stringColumn->metadata.lengths[bid];
	for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
		if (i + tid < min(entry_length, TILE_SIZE)) {
			curr_char_indices[i + tid] = i + tid;
			curr_level_match_node_idxes[i + tid] = nfa->start_state;
		}
	}
	if (tid == 0) {
		curr_level_match_node_num = min(entry_length, TILE_SIZE);
		next_level_match_node_num = 0;
		match_flag = false;
	}
	__syncthreads();
#pragma GCC ivdep //This is supposed to be equivalent to applying __restrict__ keyword to all pointers, even those as structure members.
	while (true) {		
		for (int i = 0; i < curr_level_match_node_num; i += blockDim.x) {
			if (i + tid < curr_level_match_node_num) {
				int curr_level_node_idx = curr_level_match_node_idxes[i + tid];
				for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++)
				{
					int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || stringColumn->get_char(bid, curr_char_indices[i + tid]) == match_char) {
							if (next_level_match_node_num < TILE_SIZE && (curr_char_indices[i + tid] + 1) < TILE_SIZE) {
								unsigned int bQIdx = atomicAdd(&next_level_match_node_num, 1);
								next_level_match_node_idxes[bQIdx] = next_level_node_idx;
								next_char_indices[bQIdx] = curr_char_indices[i + tid] + 1;
							}
						}
					}
				}
			}
		}
		__syncthreads();

		if (next_level_match_node_num == 0) {
			break;
		}

		for (int i = 0; i < min(min(entry_length, TILE_SIZE), next_level_match_node_num); i += blockDim.x) {
			if (i + tid < min(min(entry_length, TILE_SIZE), next_level_match_node_num)) {
				curr_level_match_node_idxes[i + tid] = next_level_match_node_idxes[i + tid];
				curr_char_indices[i + tid] = next_char_indices[i + tid];

				for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
					if (curr_level_match_node_idxes[i + tid] == nfa->end_states[end_node_idx]) {
						atomicExch(result + bid, 1);
						match_flag = true;
					}
				}
			}
		}		
		if (tid == 0) {
			curr_level_match_node_num = next_level_match_node_num;
			next_level_match_node_num = 0; // back to zero
		}
		__syncthreads();
		if (match_flag) break;
	}
}

/*
Find end state together, the whole block explores the current level together!
Add shared memory for data.
*/
__global__ void BQ_regex_gpu1(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	/*
	In this kernel, the index is for each char, which is waiting for matching,
	in the beginning, the whole entry of chars are waiting for matching.
	After the first iteration, the number should be greatly reduced.
	*/
	__shared__ bool match_flag;
	__shared__ unsigned int curr_level_match_node_num;
	__shared__ unsigned int next_level_match_node_num;
	__shared__ int curr_level_match_node_idxes[TILE_SIZE]; // start with entry_length x nfa->start_state
	__shared__ int next_level_match_node_idxes[TILE_SIZE];
	__shared__ unsigned int curr_char_indices[TILE_SIZE];
	__shared__ unsigned int next_char_indices[TILE_SIZE];
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes

	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;

	const int entry_length = stringColumn->metadata.lengths[bid];
	for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
		if (i + tid < min(entry_length, TILE_SIZE)) {
			charTile[i + tid] = stringColumn->get_char(bid, i + tid);
			curr_char_indices[i + tid] = i + tid;
			curr_level_match_node_idxes[i + tid] = nfa->start_state;
		}
	}
	if (tid == 0) {
		curr_level_match_node_num = min(entry_length, TILE_SIZE);
		next_level_match_node_num = 0;
		match_flag = false;
	}
	__syncthreads();
#pragma GCC ivdep //This is supposed to be equivalent to applying __restrict__ keyword to all pointers, even those as structure members.
	while (true) {
		for (int i = 0; i < curr_level_match_node_num; i += blockDim.x) {
			if (i + tid < curr_level_match_node_num) {
				// for (int i = 0; i < curr_level_match_node_num; i++) 
				int curr_level_node_idx = curr_level_match_node_idxes[i + tid];
				for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++)
				{
					int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || charTile[curr_char_indices[i + tid]] == match_char) {
							if (next_level_match_node_num < TILE_SIZE && (curr_char_indices[i + tid] + 1) < TILE_SIZE) {
								unsigned int bQIdx = atomicAdd(&next_level_match_node_num, 1);
								next_level_match_node_idxes[bQIdx] = next_level_node_idx;
								next_char_indices[bQIdx] = curr_char_indices[i + tid] + 1;
							}
						}
					}
				}
			}
		}
		__syncthreads();

		if (next_level_match_node_num == 0) {
			break;
		}

		for (int i = 0; i < min(min(entry_length, TILE_SIZE), next_level_match_node_num); i += blockDim.x) {
			if (i + tid < min(min(entry_length, TILE_SIZE), next_level_match_node_num)) {
				curr_level_match_node_idxes[i + tid] = next_level_match_node_idxes[i + tid];
				curr_char_indices[i + tid] = next_char_indices[i + tid];

				for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
					if (curr_level_match_node_idxes[i + tid] == nfa->end_states[end_node_idx]) {
						atomicExch(result + bid, 1);
						match_flag = true;
					}
				}
			}
		}

		if (tid == 0) {
			curr_level_match_node_num = next_level_match_node_num;
			next_level_match_node_num = 0; // back to zero
		}
		__syncthreads();
		if (match_flag) break;
	}
}

/*
Find end state together, the whole block explores the current level together!
Add shared memory for data.
*/
__global__ void BQ_regex_gpu2(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	/*
	In this kernel, the index is for each char, which is waiting for matching,
	in the beginning, the whole entry of chars are waiting for matching.
	After the first iteration, the number should be greatly reduced.
	*/
	__shared__ bool match_flag;
	__shared__ unsigned int curr_level_match_node_num;
	__shared__ unsigned int next_level_match_node_num;
	__shared__ int curr_level_match_node_idxes[TILE_SIZE]; // start with entry_length x nfa->start_state
	__shared__ int next_level_match_node_idxes[TILE_SIZE];
	__shared__ char curr_char_data[TILE_SIZE];
	__shared__ unsigned int curr_char_indices[TILE_SIZE];
	__shared__ char next_char_data[TILE_SIZE];
	__shared__ unsigned int next_char_indices[TILE_SIZE];
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;

	for (int idx_str = bid; idx_str < stringColumn->metadata.num_rows; idx_str += (gridDim.x)) {

		const int entry_length = stringColumn->metadata.lengths[idx_str];

		if (tid < min(entry_length, TILE_SIZE)) {
			charTile[tid] = stringColumn->get_char(idx_str, tid);
			curr_level_match_node_num = min(entry_length, TILE_SIZE);
			curr_char_data[tid] = stringColumn->get_char(idx_str, tid);
			curr_char_indices[tid] = tid;
			curr_level_match_node_idxes[tid] = nfa->start_state;
			next_level_match_node_num = 0;
			match_flag = false;
		}
		__syncthreads();

		while (true) {
			if (tid < curr_level_match_node_num) {
				// for (int i = 0; i < curr_level_match_node_num; i++) 
				int curr_level_node_idx = curr_level_match_node_idxes[tid];
				for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++)
				{
					int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || curr_char_data[tid] == match_char) {
							if (next_level_match_node_num < TILE_SIZE && (curr_char_indices[tid] + 1) < TILE_SIZE) {
								unsigned int bQIdx = atomicAdd(&next_level_match_node_num, 1);
								next_level_match_node_idxes[bQIdx] = next_level_node_idx;
								next_char_data[bQIdx] = charTile[curr_char_indices[tid] + 1];
								next_char_indices[bQIdx] = curr_char_indices[tid] + 1;
							}
						}
					}
				}
			}
			__syncthreads();

			if (next_level_match_node_num == 0) {
				break;
			}

			if (tid < min(min(entry_length, TILE_SIZE), next_level_match_node_num)) {
				curr_level_match_node_idxes[tid] = next_level_match_node_idxes[tid];
				curr_char_data[tid] = next_char_data[tid];
				curr_char_indices[tid] = next_char_indices[tid];
				curr_level_match_node_num = next_level_match_node_num;
			}
			__syncthreads();
			next_level_match_node_num = 0; // back to zero

			if (tid < curr_level_match_node_num) {
				for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
					if (curr_level_match_node_idxes[tid] == nfa->end_states[end_node_idx]) {
						atomicExch(result + idx_str, 1);
						match_flag = true;
					}
				}
			}
			__syncthreads();
			if (match_flag) break;
		}
	}
}


/*
Find end state together, the whole block explores the current level together!
Add shared memory for both data and nfa.
*/
__global__ void BQ_regex_gpu3(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	/*
	In this kernel, the index is for each char, which is waiting for matching,
	in the beginning, the whole entry of chars are waiting for matching.
	After the first iteration, the number should be greatly reduced.
	*/
	extern __shared__ char nfaTile[];
	__shared__ bool match_flag;
	__shared__ unsigned int curr_level_match_node_num;
	__shared__ unsigned int next_level_match_node_num;
	__shared__ int curr_level_match_node_idxes[TILE_SIZE]; // start with entry_length x nfa->start_state
	__shared__ int next_level_match_node_idxes[TILE_SIZE];
	__shared__ unsigned int curr_char_indices[TILE_SIZE];
	__shared__ unsigned int next_char_indices[TILE_SIZE];
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K/2 bytes

	const int bid = blockIdx.x;
	const int tid = threadIdx.x;

	const int nfa_num_nodes = nfa->num_nodes;
	const int nfa_num_edges = nfa->num_edges;
	const int nfa_num_match_chars = nfa->num_match_chars;
	const int nfa_num_end_states = nfa->num_of_end_states;
	const int entry_length = stringColumn->metadata.lengths[bid];
	
	int* nfa_node_outEdges_nums = (int*)&nfaTile[0];
	int* nfa_node_outEdges_offsets = (int*)&nfa_node_outEdges_nums[nfa_num_nodes];
	int* nfa_outEdges_endNode = (int*)&nfa_node_outEdges_offsets[nfa_num_nodes];
	int* nfa_outEdges_match_chars_offsets = (int*)&nfa_outEdges_endNode[nfa_num_edges];
	int* nfa_outEdges_match_chars_nums = (int*)&nfa_outEdges_match_chars_offsets[nfa_num_edges];
	int* nfa_end_states = (int*)&nfa_outEdges_match_chars_nums[nfa_num_edges]; 
	bool* nfa_outEdges_isUnconditionalTransition = (bool*)&nfa_end_states[nfa_num_end_states];
	char* nfa_outEdges_match_chars = (char*)&nfa_outEdges_isUnconditionalTransition[nfa_num_edges];
	//printf("total bytes %d\n", 2 * nfa_num_nodes * sizeof(int) + 3 * nfa_num_edges * sizeof(int) + nfa_num_edges * sizeof(bool) + nfa_num_match_chars * sizeof(char));

	if (tid < nfa_num_nodes) {
		nfa_node_outEdges_nums[tid] = nfa->node_outEdges_nums[tid];
		nfa_node_outEdges_offsets[tid] = nfa->node_outEdges_offsets[tid];
	}

	if (tid < nfa_num_edges) {
		nfa_outEdges_endNode[tid] = nfa->outEdges_endNode[tid]; //printf("### nfa_outEdges_endNode\n"); 
		nfa_outEdges_isUnconditionalTransition[tid] = nfa->outEdges_isUnconditionalTransition[tid]; //printf("### nfa_outEdges_isUnconditionalTransition\n");
		nfa_outEdges_match_chars_offsets[tid] = nfa->outEdges_match_chars_offsets[tid]; //printf("### nfa_outEdges_match_chars_offsets\n");
		nfa_outEdges_match_chars_nums[tid] = nfa->outEdges_match_chars_nums[tid]; //printf("### nfa_outEdges_match_chars_nums\n");
	}

	if (tid < nfa_num_match_chars) {
		nfa_outEdges_match_chars[tid] = nfa->outEdges_match_chars[tid]; //printf("### nfa_outEdges_match_chars\n");
	}

	if (tid < nfa_num_end_states) {
		nfa_end_states[tid] = nfa->end_states[tid];
	}


	for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
		if (i + tid < min(entry_length, TILE_SIZE)) {
			charTile[i + tid] = stringColumn->get_char(bid, i + tid);
			curr_char_indices[i + tid] = i + tid;
			curr_level_match_node_idxes[i + tid] = nfa->start_state;
		}
	}
	if (tid == 0) {
		curr_level_match_node_num = min(entry_length, TILE_SIZE);
		next_level_match_node_num = 0;
		match_flag = false;
	}
	__syncthreads();
#pragma GCC ivdep //This is supposed to be equivalent to applying __restrict__ keyword to all pointers, even those as structure members.
	while (true) {
		for (int i = 0; i < curr_level_match_node_num; i += blockDim.x) {
			if (i + tid < curr_level_match_node_num) {
				int curr_level_node_idx = curr_level_match_node_idxes[i + tid];
				for (int j = 0; j < nfa_node_outEdges_nums[curr_level_node_idx]; j++)
				{
					int next_level_node_idx = nfa_outEdges_endNode[nfa_node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa_outEdges_match_chars_nums[nfa_node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa_outEdges_match_chars[nfa_outEdges_match_chars_offsets[nfa_node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa_outEdges_isUnconditionalTransition[nfa_node_outEdges_offsets[curr_level_node_idx] + j] || charTile[curr_char_indices[i + tid]] == match_char) {
							if (next_level_match_node_num < TILE_SIZE && (curr_char_indices[i + tid] + 1) < TILE_SIZE) {
								unsigned int bQIdx = atomicAdd(&next_level_match_node_num, 1);
								next_level_match_node_idxes[bQIdx] = next_level_node_idx;
								next_char_indices[bQIdx] = curr_char_indices[i + tid] + 1;
							}
						}
					}
				}
			}
		}
		__syncthreads();

		if (next_level_match_node_num == 0) {
			break;
		}

		for (int i = 0; i < min(min(entry_length, TILE_SIZE), next_level_match_node_num); i += blockDim.x) {
			if (i + tid < min(min(entry_length, TILE_SIZE), next_level_match_node_num)) {
				curr_level_match_node_idxes[i + tid] = next_level_match_node_idxes[i + tid];
				curr_char_indices[i + tid] = next_char_indices[i + tid];

				for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
					if (curr_level_match_node_idxes[i + tid] == nfa_end_states[end_node_idx]) {
						atomicExch(result + bid, 1);
						match_flag = true;
					}
				}
			}
		}

		if (tid == 0) {
			curr_level_match_node_num = next_level_match_node_num;
			next_level_match_node_num = 0; // back to zero
		}
		__syncthreads();
		if (match_flag) break;
	}
}


#define ENTRIES_BLOCK 100
__global__ void block_coarsening_regex_gpu(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	extern __shared__ char nfaTile[];
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;

	const int nfa_start_state = nfa->start_state;
	const int nfa_num_nodes = nfa->num_nodes;
	const int nfa_num_edges = nfa->num_edges;
	const int nfa_num_match_chars = nfa->num_match_chars;
	const int nfa_num_end_states = nfa->num_of_end_states;

	int* nfa_node_outEdges_nums = (int*)&nfaTile[0];
	int* nfa_node_outEdges_offsets = (int*)&nfa_node_outEdges_nums[nfa_num_nodes];
	int* nfa_outEdges_endNode = (int*)&nfa_node_outEdges_offsets[nfa_num_nodes];
	int* nfa_outEdges_match_chars_offsets = (int*)&nfa_outEdges_endNode[nfa_num_edges];
	int* nfa_outEdges_match_chars_nums = (int*)&nfa_outEdges_match_chars_offsets[nfa_num_edges];
	int* nfa_end_states = (int*)&nfa_outEdges_match_chars_nums[nfa_num_edges];
	bool* nfa_outEdges_isUnconditionalTransition = (bool*)&nfa_end_states[nfa_num_end_states];
	char* nfa_outEdges_match_chars = (char*)&nfa_outEdges_isUnconditionalTransition[nfa_num_edges];

	if (tid < nfa_num_nodes) {
		nfa_node_outEdges_nums[tid] = nfa->node_outEdges_nums[tid];
		nfa_node_outEdges_offsets[tid] = nfa->node_outEdges_offsets[tid];
	}

	if (tid < nfa_num_edges) {
		nfa_outEdges_endNode[tid] = nfa->outEdges_endNode[tid]; //printf("### nfa_outEdges_endNode\n"); 
		nfa_outEdges_isUnconditionalTransition[tid] = nfa->outEdges_isUnconditionalTransition[tid]; //printf("### nfa_outEdges_isUnconditionalTransition\n");
		nfa_outEdges_match_chars_offsets[tid] = nfa->outEdges_match_chars_offsets[tid]; //printf("### nfa_outEdges_match_chars_offsets\n");
		nfa_outEdges_match_chars_nums[tid] = nfa->outEdges_match_chars_nums[tid]; //printf("### nfa_outEdges_match_chars_nums\n");
	}

	if (tid < nfa_num_match_chars) {
		nfa_outEdges_match_chars[tid] = nfa->outEdges_match_chars[tid]; //printf("### nfa_outEdges_match_chars\n");
	}

	if (tid < nfa_num_end_states) {
		nfa_end_states[tid] = nfa->end_states[tid];
	}


	for (int num_entries = 0; num_entries < ENTRIES_BLOCK && bid * ENTRIES_BLOCK + num_entries < stringColumn->metadata.num_rows; num_entries++) {
		const int entry_length = stringColumn->metadata.lengths[bid * ENTRIES_BLOCK + num_entries];
		for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
			if (i + tid < min(entry_length, TILE_SIZE))
				charTile[i + tid] = stringColumn->get_char(bid * ENTRIES_BLOCK + num_entries, i + tid);
		}
		__syncthreads();

		for (int t = 0; t < min(entry_length, TILE_SIZE); t += blockDim.x) {
			if ((t + tid) < min(entry_length, TILE_SIZE)) {
				int curr_level_match_node_num = 0;
				int curr_level_match_node_idxes[BUFFER_SIZE];
				int next_level_match_node_num = 0;
				int next_level_match_node_idxes[BUFFER_SIZE];
				const int num_chars_for_search = min(entry_length - (t + tid), TILE_SIZE - (t + tid));
				int idx = 0;
				curr_level_match_node_idxes[curr_level_match_node_num++] = nfa_start_state;

				while (idx < num_chars_for_search) {
					char this_char = charTile[(t + tid) + idx];
					for (int i = 0; i < curr_level_match_node_num; i++) { // curr_level_node_idx: curr_level_match_node_idxes
						int curr_level_node_idx = curr_level_match_node_idxes[i];
						for (int j = 0; j < nfa_node_outEdges_nums[curr_level_node_idx]; j++) // edge_pair : curr_level_node.outEdges
						{
							int next_level_node_idx = nfa_outEdges_endNode[nfa_node_outEdges_offsets[curr_level_node_idx] + j];
							for (int k = 0; k < nfa_outEdges_match_chars_nums[nfa_node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
							{
								char match_char = nfa_outEdges_match_chars[nfa_outEdges_match_chars_offsets[nfa_node_outEdges_offsets[curr_level_node_idx] + j] + k];
								if (nfa_outEdges_isUnconditionalTransition[nfa_node_outEdges_offsets[curr_level_node_idx] + j] || this_char == match_char) {
									if (next_level_match_node_num < BUFFER_SIZE)
										next_level_match_node_idxes[next_level_match_node_num++] = next_level_node_idx;
								}
							}
						}
					}

					if (next_level_match_node_num == 0) {
						break;
					}
					else {
						idx++;
						for (int i = 0; i < next_level_match_node_num; i++) {// assign next to next iteration
							curr_level_match_node_idxes[i] = next_level_match_node_idxes[i];
						}
						curr_level_match_node_num = next_level_match_node_num;
						next_level_match_node_num = 0; // back to zero
						bool match_flag = false;
						for (int final_node_idx = 0; final_node_idx < curr_level_match_node_num; final_node_idx++) {
							for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
								if (curr_level_match_node_idxes[final_node_idx] == nfa_end_states[end_node_idx]) {
									atomicExch(result + bid * ENTRIES_BLOCK + num_entries, 1);
									match_flag = true;
								}
							}
						}
						if (match_flag) break;
					}
				}
			}
		}
	}
}


__global__ void dynamic_para_regex_gpu_child(MyStringColumn* stringColumn, MyGPUNFA* nfa, const int entry_id, const int entry_length, int* result) {
	__shared__ char charTile[TILE_SIZE]; // assume the size of is less than 1K bytes
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	const int nfa_num_end_states = nfa->num_of_end_states;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("entry_id %d", entry_id);
		printf("entry_length %d", entry_length);
	}
		

	for (int i = blockIdx.x * blockDim.x; i < TILE_SIZE; i += blockDim.x) {
		if (i + threadIdx.x < min(entry_length, TILE_SIZE))
			charTile[i + threadIdx.x- blockIdx.x * blockDim.x] = stringColumn->get_char(entry_id, i + threadIdx.x);
	}
	__syncthreads();

	if (t < min(entry_length, TILE_SIZE)) {
		int curr_level_match_node_num = 0;
		int curr_level_match_node_idxes[BUFFER_SIZE];
		int next_level_match_node_num = 0;
		int next_level_match_node_idxes[BUFFER_SIZE];
		const int num_chars_for_search = min(entry_length - t, TILE_SIZE - t);
		int idx = 0;
		curr_level_match_node_idxes[curr_level_match_node_num++] = nfa->start_state;

		while (idx < num_chars_for_search) {
			char this_char = charTile[t + idx- blockIdx.x * blockDim.x];
			for (int i = 0; i < curr_level_match_node_num; i++) { // curr_level_node_idx: curr_level_match_node_idxes
				int curr_level_node_idx = curr_level_match_node_idxes[i];
				for (int j = 0; j < nfa->node_outEdges_nums[curr_level_node_idx]; j++) // edge_pair : curr_level_node.outEdges
				{
					int next_level_node_idx = nfa->outEdges_endNode[nfa->node_outEdges_offsets[curr_level_node_idx] + j];
					for (int k = 0; k < nfa->outEdges_match_chars_nums[nfa->node_outEdges_offsets[curr_level_node_idx] + j]; k++) //  match_char : edge_pair.second.match_chars
					{
						char match_char = nfa->outEdges_match_chars[nfa->outEdges_match_chars_offsets[nfa->node_outEdges_offsets[curr_level_node_idx] + j] + k];
						if (nfa->outEdges_isUnconditionalTransition[nfa->node_outEdges_offsets[curr_level_node_idx] + j] || this_char == match_char) {
							if (next_level_match_node_num < BUFFER_SIZE)
								next_level_match_node_idxes[next_level_match_node_num++] = next_level_node_idx;
						}
					}
				}
			}

			if (next_level_match_node_num == 0) {
				break;
			}
			else {
				idx++;
				for (int i = 0; i < next_level_match_node_num; i++) {// assign next to next iteration
					curr_level_match_node_idxes[i] = next_level_match_node_idxes[i];
				}
				curr_level_match_node_num = next_level_match_node_num;
				next_level_match_node_num = 0; // back to zero
				for (int final_node_idx = 0; final_node_idx < curr_level_match_node_num; final_node_idx++) {
					for (int end_node_idx = 0; end_node_idx < nfa_num_end_states; end_node_idx++) {
						if (curr_level_match_node_idxes[final_node_idx] == nfa->end_states[end_node_idx]) {
							atomicExch(result, 1);
						}
					}
				}
				if (*result>0) break;
			}
		}
	}
}

__global__ void dynamic_para_regex_gpu(MyStringColumn* stringColumn, MyGPUNFA* nfa, int* result) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < stringColumn->metadata.num_rows) {
		const int entry_length = stringColumn->metadata.lengths[j];

		//cudaStream_t stream;
		//cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (entry_length < NUM_THREADS) {
			dynamic_para_regex_gpu_child << <1, entry_length >> > (stringColumn, nfa, j, entry_length, &result[j]);
		}
		else {
			dynamic_para_regex_gpu_child << <(entry_length - 1)/ NUM_THREADS+1, NUM_THREADS >> > (stringColumn, nfa, j,entry_length, &result[j]);
		}
		//cudaDeviceSynchronize();
		//cudaStreamDestroy(stream);
	}
}