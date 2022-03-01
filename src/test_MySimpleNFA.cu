#include "MySimpleNFA.h"
#include "helper_string.h"

void printHelp(void)
{
    printf("Usage: test_loadNFA.cu --file= \n");
    printf("file needs to be the ABSOLUTE PATH TO THE FILE\n");
    printf("example: ./build/exe/src/test_loadNFA.cu.exe --file=/home/kwu/ece508project/datasets/literally_libfsm.libfsm_output\n");
}

int main(const int argc, const char *argv[])
{
    if (!checkCmdLineFlag(argc, argv, "file"))
    {
        printf("Error: file option not found\n");
        printHelp();
        return 1;
    }
    char *filename = NULL;
    getCmdLineArgumentString(argc, argv, "file", &filename);
    MySimpleNFA test_nfa1 = LoadNFAFromLibFSMOutput(filename);
    test_nfa1.printTransitions();
}