#include "NFALoader.h"
#include "helper_string.h"
#include <iostream>

void printHelp(void) {
    printf("Usage: test_loadNFA.cu --mode= --file= \n");
    printf("mode can either be anml or mynfa\n");
    printf("file needs to be the ABSOLUTE PATH TO THE FILE\n");
    printf("example: ./build/exe/src/test_loadNFA.cu.exe --mode=anml --file=/home/kwu/ece508project/datasets/ANMLZoo/Bro217/anml/bro217.anml\n");
}

int main(const int argc, const char *argv[]) {
    //if (!checkCmdLineFlag(argc, argv, "mode") || !checkCmdLineFlag(argc, argv, "file")) {
    //    printf("Error: mode or flag option not found\n");
    //    printHelp();
    //    return 1;
    //}
    char* mode = NULL; char* filename = NULL;
    /*getCmdLineArgumentString(argc, argv, "mode", &mode);
    getCmdLineArgumentString(argc, argv, "file", &filename);*/
    mode = "anml";
    filename = "C:/Work/High_Perfomance_Multiphysics_Simulation_CEM/UIUC_Courses/ECE 508/PROJECT/ANMLZoo/Bro217/anml/bro217.anml";
    if (strcmp(mode,"anml")&&strcmp(mode,"mynfa")){
        printf("Error: mode must be either anml or mynfa\n");
        printHelp();
        return 1;
    }
    if (!strcmp(mode,"anml")){
        NFA* test_nfa1  = load_nfa_from_anml(filename);
        std::cout << "test_nfa1->size() " << test_nfa1->size() << std::endl;
        //test_nfa1->print();
        std::cout << "test_nfa1->get_indegree_of_node(\"__260__\") " << test_nfa1->get_indegree_of_node("__260__") << std::endl;
        std::cout << "test_nfa1->get_outdegree_of_node(\"__260__\") " << test_nfa1->get_outdegree_of_node("__260__") << std::endl; 
        std::cout << "test_nfa1->get_int_id_by_str_id(\"__260__\") " << test_nfa1->get_int_id_by_str_id("__260__") << std::endl;
        std::cout << "test_nfa1->get_adj(\"__260__\") " << test_nfa1->get_adj("__260__").at(0) << std::endl;
        vector<string> str_ids;
        str_ids = test_nfa1->get_adj("__359__"); // next nodes
        int level = 0;
        while (str_ids.size() != 0) {
            for (auto str_id : str_ids) {
                auto from_str_ids = test_nfa1->get_from(str_id); // previous nodes
                Node* str_node = test_nfa1->get_node_by_str_id(str_id); // node pointer
                
                std::cout << "From: " << from_str_ids.at(0) << std::endl;
                std::cout << "Level " << level << " : " << str_id << " : " << str_node->symbol_set_str.at(1) << std::endl;
            }
            str_ids = test_nfa1->get_adj(str_ids.at(0));   
            level++;
        }
        
        std::cout << "test_nfa1->get_node_by_int_id(1) " << test_nfa1->get_node_by_int_id(1) << std::endl;
        std::cout << "test_nfa1->get_node_by_int_id(2) " << test_nfa1->get_node_by_int_id(2) << std::endl;

        printf("Success: anml loading scheme implemented\n");
    }
    else{
        printf("Error: mynfa loading scheme not implemented\n");
        return 1;
    }
    

}