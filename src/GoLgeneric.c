#include <string.h>
#include <x86intrin.h>
#include <stdio.h>

#include "GoLgeneric.h"

void printhelp(){
    printf("This executable accepts Game of Life input parameters.\n"
        "All parameters default to 0.\n"
        "These are:\n"
        "\t-w [width]\n"
        "\t-h [height]\n"
        "\t-N [width/heigt] Specifies a N for both width and height (square)\n"
        "\t-g [density] (double)Generates map based on -w and -h with specified density.\n"
        "\t-s [steps]   Tells the program to simulate x steps\n"
        "\t-t [seconds] (double)Tells the program to simulate for x seconds\n"
        "\t-l           Simulate till all life is still/repeating\n"
        "\t-r [csv file]Specifies the file to report to.\n"
        "\t--seed [seed]Specifies a seed for generation of the map\n");
}

int parseInput(int nargs, char ** args, struct inputParameters* inpp){
    memset(inpp, 0, sizeof(inpp));
    
    if (nargs == 2 && (strcmp(args[1], "--help")==0 || strcmp(args[1], "-h"))) {
        printhelp();
    }
    
    //Extract arguments
    int arg = 1;
    
    while(arg < nargs) {
        char *cmd = args[arg++];
        
        if (strcmp(cmd, "-w")==0)
           inpp->width = atoll(args[arg++]);
        if (strcmp(cmd, "-h")==0)
            inpp->height = atoll(args[arg++]);
        if (strcmp(cmd, "-d")==0)
            inpp->density = atof(args[arg++]);
        if (strcmp(cmd, "-s")==0)
            inpp->steps = atoll(args[arg++]);
        if (strcmp(cmd, "-l")==0)
            inpp->simulateTillStill = atoi(args[arg++]);
        if (strcmp(cmd, "-t")==0)
            inpp->time = atof(args[arg++]);
        if (strcmp(cmd, "r")==0)
            strcpy(inpp->reportfile,args[arg++]);
        if (strcmp(cmd, "--seed")==0)
            inpp->seed = atol(args[arg++]);
        if (strcmp(cmd, "-N")==0)
            inpp->width = inpp->height = atoll(args[arg++]);
    }
    
}

//Integer map opterations
int generateMapInt(int *map, double density, unsigned long seed, struct vector2u size) {
    unsigned int s = seed;

    size_t elems = size.x * size.y;
    size_t i;
    
    //No multithead for consistancy reasons.
    for(i=0; i<elems; i+=4) {
            map[i+0] = (int)density>=((double)rand_r(&s)/(double)RAND_MAX);
            map[i+1] = (int)density>=((double)rand_r(&s)/(double)RAND_MAX);
            map[i+2] = (int)density>=((double)rand_r(&s)/(double)RAND_MAX);
            map[i+3] = (int)density>=((double)rand_r(&s)/(double)RAND_MAX);
    }
    for(i-=3; i<elems; i++) {
        map[i+0] = (int)density>=((double)rand_r(&s)/(double)RAND_MAX);
    }
}
size_t countAliveInt(int *map, struct vector2u size) {
    size_t elems = size.x * size.y;
    size_t i, acc1, acc2, acc3, acc4;
    acc1 = acc2 = acc3 = acc4 = 0;
    //No multithead for consistancy reasons.
    for(i=0; i<elems; i+=4) {
        acc1 += map[i+0];
        acc2 += map[i+1];
        acc3 += map[i+2];
        acc4 += map[i+3];
    }
    for(i-=3; i<elems; i++)
        acc1 += map[i];
}


size_t countAliveIntSSE(int *pDatapointer, struct vector2u size) {
    //Meh
}

//binary map operations
int generateMapBinary(void *pDatapointer, double density,unsigned long seed, struct vector2u size);
size_t countAliveBinary(void *pDatapointer, struct vector2u size);