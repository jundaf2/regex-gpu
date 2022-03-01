#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <unordered_map>
#include "char_string_int_conversion.h"
#include "utils.h"

class MySimpleNode;
class MySimpleEdge
{
public:
    std::vector<char> match_chars{};
    int endNode;
    bool isUnconditionalTransition;
    MySimpleEdge()
    {
    }
    MySimpleEdge(int endNode, bool isUnconditionalTransition)
    {
        this->endNode = endNode;
        this->isUnconditionalTransition = isUnconditionalTransition;
    }
};

class MySimpleNode
{
public:
    std::unordered_map<int, MySimpleEdge> outEdges{}; //node_idx, MySimpleEdge pair
    int node_idx;
    MySimpleNode(int node_idx)
    {
        this->node_idx = node_idx;
    }
};

class MySimpleNFA
{
public:
    std::vector<int> end_states{};
    int start_state;
    std::vector<MySimpleNode> nodes{};
    MySimpleNFA()
    {
    }
    void replenish_nodes(int count)
    {
        for (int node_to_add_loopidx = 0; node_to_add_loopidx < count; node_to_add_loopidx++)
        {
            nodes.push_back(MySimpleNode(nodes.size()));
        }
    }
    void printTransitions()
    {
        std::cout << "Start node: " << start_state << std::endl;
        std::cout << "End node: ";
        for (int end_idx = 0; end_idx < end_states.size(); end_idx++)
        {
            std::cout << end_states[end_idx] << ", ";
        }
        std::cout << std::endl;
        for (int node_idx = 0; node_idx < nodes.size(); node_idx++)
        {

            std::cout << "Node " << node_idx << ": ";
            for (auto edge_pair : nodes[node_idx].outEdges)
            {
                std::cout << "(" << edge_pair.first << ",";
                std::cout << edge_pair.second.isUnconditionalTransition << ") ";
                assert(edge_pair.second.endNode == edge_pair.first);
                std::cout << "( ";
                for (auto match_char : edge_pair.second.match_chars)
                {
                    std::cout << get_escpaed_string_from_char(match_char) << ",";
                }
                std::cout << ") " << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

class MyGPUNFA {
public:
    int start_state;
    int num_of_end_states;
    int* end_states;

    // Nodes
    int num_nodes;
    int* __restrict__ node_outEdges_offsets;
    int* __restrict__ node_outEdges_nums;

    int num_edges;
    int* __restrict__ outEdges_endNode;
    bool* __restrict__ outEdges_isUnconditionalTransition;

    int num_match_chars;
    int* __restrict__ outEdges_match_chars_offsets;
    int* __restrict__ outEdges_match_chars_nums;
    char*__restrict__ outEdges_match_chars;

    int total_num_of_bytes;

    MyGPUNFA() = default;
    MyGPUNFA(MySimpleNFA simple_nfa) {
        this->start_state = simple_nfa.start_state;
        this->num_of_end_states = simple_nfa.end_states.size();
        this->num_nodes = simple_nfa.nodes.size();

        cuda_err_chk(cudaMalloc((void**)&this->end_states, this->num_of_end_states * sizeof(int)));
        cuda_err_chk(cudaMalloc((void**)&this->node_outEdges_offsets, this->num_nodes * sizeof(int)));
        cuda_err_chk(cudaMalloc((void**)&this->node_outEdges_nums, this->num_nodes * sizeof(int)));

        // node to edge
        int* node_outEdges_offsets_h;
        int* node_outEdges_nums_h;

        int* outEdges_endNode_h;
        bool* outEdges_isUnconditionalTransition_h;
        int* outEdges_match_chars_offsets_h;
        int* outEdges_match_chars_nums_h;

        int edge_offset = 0;
        int match_char_offset = 0;
        node_outEdges_offsets_h = (int*)malloc(this->num_nodes * sizeof(int));
        node_outEdges_nums_h = (int*)malloc(this->num_nodes * sizeof(int));
        for (int i = 0; i < this->num_nodes; i++) {
            node_outEdges_offsets_h[i] = edge_offset;
            node_outEdges_nums_h[i] = simple_nfa.nodes[i].outEdges.size();
            edge_offset += simple_nfa.nodes[i].outEdges.size();
        }

        this->num_edges = edge_offset;
        cuda_err_chk(cudaMalloc((void**)&this->outEdges_endNode, edge_offset * sizeof(int)));
        cuda_err_chk(cudaMalloc((void**)&this->outEdges_isUnconditionalTransition, edge_offset * sizeof(bool)));
        cuda_err_chk(cudaMalloc((void**)&this->outEdges_match_chars_offsets, edge_offset * sizeof(int)));
        cuda_err_chk(cudaMalloc((void**)&this->outEdges_match_chars_nums, edge_offset * sizeof(int)));
        outEdges_endNode_h = (int*)malloc(edge_offset * sizeof(int));
        outEdges_isUnconditionalTransition_h = (bool*)malloc(edge_offset * sizeof(bool));
        outEdges_match_chars_offsets_h = (int*)malloc(edge_offset * sizeof(int));
        outEdges_match_chars_nums_h = (int*)malloc(edge_offset * sizeof(int));

        for (int i = 0; i < this->num_nodes; i++) {
            int j = 0;
            for (auto iter = simple_nfa.nodes[i].outEdges.begin(); iter != simple_nfa.nodes[i].outEdges.end(); ++iter) {
                outEdges_endNode_h[node_outEdges_offsets_h[i] + j] = iter->second.endNode;
                outEdges_isUnconditionalTransition_h[node_outEdges_offsets_h[i] + j] = iter->second.isUnconditionalTransition;

                outEdges_match_chars_offsets_h[node_outEdges_offsets_h[i] + j] = match_char_offset;
                outEdges_match_chars_nums_h[node_outEdges_offsets_h[i] + j] = iter->second.match_chars.size();
                match_char_offset += iter->second.match_chars.size();
                j++;
            }
                
        }
        this->num_match_chars = match_char_offset;
        cuda_err_chk(cudaMemcpy(this->end_states, &simple_nfa.end_states[0], this->num_of_end_states * sizeof(int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->node_outEdges_offsets, node_outEdges_offsets_h, this->num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->node_outEdges_nums, node_outEdges_nums_h, this->num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->outEdges_endNode, outEdges_endNode_h, edge_offset * sizeof(int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->outEdges_isUnconditionalTransition, outEdges_isUnconditionalTransition_h, edge_offset * sizeof(bool), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->outEdges_match_chars_offsets, outEdges_match_chars_offsets_h, edge_offset * sizeof(int), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(this->outEdges_match_chars_nums, outEdges_match_chars_nums_h, edge_offset * sizeof(int), cudaMemcpyHostToDevice));

        //  edge to match_chars
        char* outEdges_match_chars_h;
        cuda_err_chk(cudaMalloc((void**)&this->outEdges_match_chars, match_char_offset * sizeof(char)));
        outEdges_match_chars_h = (char*)malloc(match_char_offset * sizeof(char));
        for (int i = 0; i < this->num_nodes; i++) {
            int j = 0;
            for (auto iter = simple_nfa.nodes[i].outEdges.begin(); iter != simple_nfa.nodes[i].outEdges.end(); ++iter) {
                for (int k = 0; k < outEdges_match_chars_nums_h[node_outEdges_offsets_h[i] + j]; k++) { // node_outEdges_offsets_h[i] + j
                    outEdges_match_chars_h[outEdges_match_chars_offsets_h[node_outEdges_offsets_h[i] + j] + k] = iter->second.match_chars[k];
                }
                j++;
            }
        }
        cuda_err_chk(cudaMemcpy(this->outEdges_match_chars, outEdges_match_chars_h, match_char_offset * sizeof(char), cudaMemcpyHostToDevice));

        // free host arrays
        free(node_outEdges_offsets_h);
        free(node_outEdges_nums_h);
        free(outEdges_endNode_h);
        free(outEdges_isUnconditionalTransition_h);
        free(outEdges_match_chars_offsets_h);
        free(outEdges_match_chars_nums_h);
        free(outEdges_match_chars_h);
        this->total_num_of_bytes = 1 * this->num_of_end_states * sizeof(int) + 2 * this->num_nodes * sizeof(int) + 3 * this->num_edges * sizeof(int) + 1 * this->num_edges * sizeof(bool) + 1 * this->num_match_chars * sizeof(char);
        std::cout << "Before " << this->total_num_of_bytes << std::endl;
        if (this->total_num_of_bytes % 128 != 0) {
            this->total_num_of_bytes = (this->total_num_of_bytes / 128 + 1) * 128;
            std::cout << "After " << this->total_num_of_bytes << std::endl;
        }
    }

    void free_() {
        cuda_err_chk(cudaFree(end_states));
        cuda_err_chk(cudaFree(node_outEdges_offsets));
        cuda_err_chk(cudaFree(node_outEdges_nums));
        cuda_err_chk(cudaFree(outEdges_endNode));
        cuda_err_chk(cudaFree(outEdges_isUnconditionalTransition));
        cuda_err_chk(cudaFree(outEdges_match_chars_offsets));
        cuda_err_chk(cudaFree(outEdges_match_chars_nums));
        cuda_err_chk(cudaFree(outEdges_match_chars));
    }

    static struct MyGPUNFA* CopyToDevice(const struct MyGPUNFA& nfa) {
        struct MyGPUNFA* gpu_nfa_ptr;
        cuda_err_chk(cudaMalloc(&gpu_nfa_ptr, sizeof(struct MyGPUNFA)));
        cuda_err_chk(cudaMemcpy(gpu_nfa_ptr, &nfa, sizeof(struct MyGPUNFA), cudaMemcpyHostToDevice));
        return gpu_nfa_ptr;
    }
};

//MySimpleNFA LoadNFAFromLibFSMOutput(const char *filename)
//{
//    // asserting the output is of 'fsm' format, i.e.,  ./re -pl fsm ...
//    // here the fsm format does not necessarily mean it is finate-state machine. Instead, it may contain epsilon transitions as some of our test cases suggest.
//    MySimpleNFA result;
//    std::ifstream input(filename);
//    for (std::string line; std::getline(input, line);)
//    {
//        if (line.find("start:") != std::string::npos)
//        {                                                //start state
//        assert(line.find(",") == std::string::npos); //assuming only one start state
//        int state_idx = std::stoi(line.substr(line.find("start:") + 6));
//        std::cout << "start state: " << state_idx << std::endl;
//        result.start_state = state_idx;
//        }
//        else if (line.find("end:") != std::string::npos)
//        { //acceptance states
//        std::string::size_type rest_str_pos;
//        int end_idx = std::stoi(line.substr(line.find("end:") + 4), &rest_str_pos);
//        std::cout << "acceptance state: " << end_idx << std::endl;
//        result.end_states.push_back(end_idx);
//        while (line.substr(rest_str_pos).find(",") != std::string::npos)
//        {
//            std::string::size_type additional_rest_substr_pos;
//            int end_idx_rest = std::stoi(line.substr(rest_str_pos).substr(line.substr(rest_str_pos).find(",") + 1), &additional_rest_substr_pos);
//            rest_str_pos += (line.substr(rest_str_pos).find(",") + 1 + additional_rest_substr_pos);
//            std::cout << "acceptance state: " << end_idx_rest << std::endl;
//            result.end_states.push_back(end_idx_rest);
//        }
//        }
//    }
//
//    std::ifstream input1(filename);
//    for (std::string line; std::getline(input1, line);)
//    {
//        if (line.find("->") != std::string::npos)
//        {
//            std::string::size_type rest_str_pos;
//            int right_hand_state_idx = std::stoi(line.substr(line.find("->") + 2), &rest_str_pos);
//            rest_str_pos += (line.find("->") + 2);
//            int left_hand_state_idx = std::stoi(line);
//            if (/*right_hand_state_idx != left_hand_state_idx &&*/ right_hand_state_idx != result.start_state) {
//                if (line.find("\"") == std::string::npos || line.find("\"") > line.find(";"))
//                {
//                    // no specified label for this transition, could be either
//                    // 0 -> 2 ?; or 0 -> 2;
//                    // we skip the first case as we don't know why this is useful
//                    // the second case means an epsilon transition
//                    if (line.find("?;") != std::string::npos && line.find("?;") < line.find(";"))
//                    {
//
//                        // more details: https://sourcegraph.com/github.com/katef/libfsm@8ec7f51b537a1e75851a2aa216c0c1e33aab4966/-/blob/src/libfsm/print/fsm.c?L180
//                        // ?; means /./ in dot, which means every symbol is accepted according to https://github.com/katef/libfsm/blob/main/doc/tutorial/re.md
//                        //std::cout << "skipping line " << line << std::endl;
//                        //std::cout << "/./ transition: " << line << std::endl;
//                        int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
//                        result.replenish_nodes(maximal_number_nodes_to_add + 1);
//                        assert(result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end());
//                        MySimpleEdge temp(right_hand_state_idx, false);
//                        result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
//                        for (int c_idx = 0; c_idx < 256; c_idx++)
//                        {
//                            char c = ascii_int_to_char(c_idx);
//                            result.nodes[left_hand_state_idx].outEdges[right_hand_state_idx].match_chars.push_back(c);
//                        }
//                        continue;
//                    }
//                    else if (line.substr(rest_str_pos) == ";")
//                    {
//                        //std::cout << "epsilon transition: " << line << std::endl;
//                        int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
//                        result.replenish_nodes(maximal_number_nodes_to_add + 1);
//                        assert(result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end());
//                        MySimpleEdge temp(right_hand_state_idx, true);
//                        result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
//                        continue;
//                    }
//                }
//                //std::cout<<"diagnosis: "<< line.find("\"")<<" " << line.find("\";")<<std::endl;
//                std::string label_str = line.substr(line.find("\"") + 1, line.find("\";") - (line.find("\"") + 1));
//                if (line.find("\";\";") != std::string::npos)
//                {
//                    // this is a negated transition
//                    label_str = ";";
//                }
//                char label_char;
//                if (label_str.length() == 1)
//                {
//                    label_char = label_str[0];
//                }
//                else if (label_str.length() == 2)
//                {
//                    switch (label_str[1])
//                    {
//                    case 't':
//                        label_char = 9;
//                        break;
//                    case 'n':
//                        label_char = 10;
//                        break;
//                    case 'v':
//                        label_char = 11;
//                        break;
//                    case 'f':
//                        label_char = 12;
//                        break;
//                    case 'r':
//                        label_char = 13;
//                        break;
//                    case '"':
//                        label_char = 34;
//                        break;
//                    case '\\':
//                        label_char = 92;
//                        break;
//                    default:
//                        std::cout << "unrecognized escape sequence: " << label_str << std::endl;
//                        assert(0 && "unrecognized escape sequence");
//                    }
//                }
//                else if (label_str.length() == 4)
//                {
//                    label_char = std::stoi(label_str.substr(2), nullptr, 16);
//                }
//                else
//                {
//                    std::cout << "unrecognized label: " << label_str << std::endl;
//                    assert(0 && "unrecognized escape sequence");
//                }
//                std::cout << "( " << left_hand_state_idx << ", " << right_hand_state_idx << ", " << label_char << "," << label_str.length() << " )" << std::endl;
//                //std::cout << line << std::endl;
//                int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
//                result.replenish_nodes(maximal_number_nodes_to_add + 1);
//                if (result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end())
//                {
//                    MySimpleEdge temp(right_hand_state_idx, false);
//                    result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
//                }
//                result.nodes[left_hand_state_idx].outEdges[right_hand_state_idx].match_chars.push_back(label_char);
//            }
//        }
//        
//    }
//    return result;
//}

MySimpleNFA LoadNFAFromLibFSMOutput(const char* filename, bool is_verbose = false)
{
    // asserting the output is of 'fsm' format, i.e.,  ./re -pl fsm ...
    // here the fsm format does not necessarily mean it is finate-state machine. Instead, it may contain epsilon transitions as some of our test cases suggest.
    MySimpleNFA result;
    std::ifstream input(filename);
    for (std::string line; std::getline(input, line);)
    {
        if (line.find("->") != std::string::npos)
        {
            std::string::size_type rest_str_pos;
            int right_hand_state_idx = std::stoi(line.substr(line.find("->") + 2), &rest_str_pos);
            rest_str_pos += (line.find("->") + 2);
            int left_hand_state_idx = std::stoi(line);
            if (line.find("\"") == std::string::npos || line.find("\"") > line.find(";"))
            {
                // no specified label for this transition, could be either
                // 0 -> 2 ?; or 0 -> 2;
                // we skip the first case as we don't know why this is useful
                // the second case means an epsilon transition
                if (line.find("?;") != std::string::npos && line.find("?;") < line.find(";"))
                {
                    // more details: https://sourcegraph.com/github.com/katef/libfsm@8ec7f51b537a1e75851a2aa216c0c1e33aab4966/-/blob/src/libfsm/print/fsm.c?L180
                    // ?; means /./ in dot, which means every symbol is accepted according to https://github.com/katef/libfsm/blob/main/doc/tutorial/re.md
                    //std::cout << "skipping line " << line << std::endl;
                    if (is_verbose)
                        std::cout << "/./ transition: " << line << std::endl;
                    int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
                    result.replenish_nodes(maximal_number_nodes_to_add + 1);
                    assert(result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end());
                    MySimpleEdge temp(right_hand_state_idx, false);
                    result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
                    for (int c_idx = 0; c_idx < 256; c_idx++)
                    {
                        char c = ascii_int_to_char(c_idx);
                        result.nodes[left_hand_state_idx].outEdges[right_hand_state_idx].match_chars.push_back(c);
                    }
                    continue;
                }
                else if (line.substr(rest_str_pos) == ";")
                {
                    if (is_verbose)
                        std::cout << "epsilon transition: " << line << std::endl;
                    int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
                    result.replenish_nodes(maximal_number_nodes_to_add + 1);
                    assert(result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end());
                    MySimpleEdge temp(right_hand_state_idx, true);
                    result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
                    continue;
                }
            }
            //std::cout<<"diagnosis: "<< line.find("\"")<<" " << line.find("\";")<<std::endl;
            std::string label_str = line.substr(line.find("\"") + 1, line.find("\";") - (line.find("\"") + 1));
            if (line.find("\";\";") != std::string::npos)
            {
                // this is a negated transition
                label_str = ";";
            }
            char label_char;
            if (label_str.length() == 1)
            {
                label_char = label_str[0];
            }
            else if (label_str.length() == 2)
            {
                switch (label_str[1])
                {
                case 't':
                    label_char = 9;
                    break;
                case 'n':
                    label_char = 10;
                    break;
                case 'v':
                    label_char = 11;
                    break;
                case 'f':
                    label_char = 12;
                    break;
                case 'r':
                    label_char = 13;
                    break;
                case '"':
                    label_char = 34;
                    break;
                case '\\':
                    label_char = 92;
                    break;
                default:
                    if (is_verbose)
                        std::cout << "unrecognized escape sequence: " << label_str << std::endl;
                    assert(0 && "unrecognized escape sequence");
                }
            }
            else if (label_str.length() == 4)
            {
                label_char = std::stoi(label_str.substr(2), nullptr, 16);
            }
            else
            {
                if (is_verbose)
                    std::cout << "unrecognized label: " << label_str << std::endl;
                assert(0 && "unrecognized escape sequence");
            }
            if (is_verbose) {
                std::cout << "( " << left_hand_state_idx << ", " << right_hand_state_idx << ", " << label_char << "," << label_str.length() << " )" << std::endl;
                std::cout << line << std::endl;
            }
            int maximal_number_nodes_to_add = max(left_hand_state_idx, right_hand_state_idx) - result.nodes.size();
            result.replenish_nodes(maximal_number_nodes_to_add + 1);
            if (result.nodes[left_hand_state_idx].outEdges.find(right_hand_state_idx) == result.nodes[left_hand_state_idx].outEdges.end())
            {
                MySimpleEdge temp(right_hand_state_idx, false);
                result.nodes[left_hand_state_idx].outEdges.emplace(std::make_pair(right_hand_state_idx, temp));
            }
            result.nodes[left_hand_state_idx].outEdges[right_hand_state_idx].match_chars.push_back(label_char);
        }
        else if (line.find("start") != std::string::npos)
        {                                                //start state
            assert(line.find(",") == std::string::npos); //assuming only one start state
            int state_idx = std::stoi(line.substr(line.find("start:") + 6));
            if (is_verbose)
                std::cout << "start state: " << state_idx << std::endl;
            result.start_state = state_idx;
        }
        else if (line.find("end:") != std::string::npos)
        { //acceptance states
            std::string::size_type rest_str_pos;
            int end_idx = std::stoi(line.substr(line.find("end:") + 4), &rest_str_pos);
            if (is_verbose)
                std::cout << "acceptance state: " << end_idx << std::endl;
            result.end_states.push_back(end_idx);
            while (line.substr(rest_str_pos).find(",") != std::string::npos)
            {
                std::string::size_type additional_rest_substr_pos;
                int end_idx_rest = std::stoi(line.substr(rest_str_pos).substr(line.substr(rest_str_pos).find(",") + 1), &additional_rest_substr_pos);
                rest_str_pos += (line.substr(rest_str_pos).find(",") + 1 + additional_rest_substr_pos);
                if (is_verbose)
                    std::cout << "acceptance state: " << end_idx_rest << std::endl;
                result.end_states.push_back(end_idx_rest);
            }
        }
    }
    return result;
}