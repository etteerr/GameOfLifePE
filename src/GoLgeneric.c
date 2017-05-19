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
        "\t-d [density] (double)Generates map based on -w and -h with specified density.\n"
        "\t-s [steps]   Tells the program to simulate x steps\n"
        "\t-t [seconds] (double)Tells the program to simulate for x seconds\n"
        "\t-l           Simulate till all life is still/repeating\n"
        "\t-r [csv file]Specifies the file to report to.\n"
        "\t--seed [seed]Specifies a seed for generation of the map\n");
}

void printInputParameters(struct inputParameters * p){
    printf("This executable accepts Game of Life input parameters.\n"
        "All parameters default to 0.\n"
        "These are:\n"
        "\t-w [%zu]\n"
        "\t-h [%zu]\n"
        "\t-N [%zu/%zu] Specifies a N for both width and height (square)\n"
        "\t-d [%e] (double)Generates map based on -w and -h with specified density.\n"
        "\t-s [%zu]   Tells the program to simulate x steps\n"
        "\t-t [%e] (double)Tells the program to simulate for x seconds\n"
        "\t-l           Simulate till all life is still/repeating\n"
        "\t-r [%s]Specifies the file to report to.\n"
        "\t--seed [%ul]Specifies a seed for generation of the map\n",
            p->width,
            p->height,
            p->width, p->height,
            p->density,
            p->steps,
            p->time,
            p->reportfile,
            p->seed);
}

int parseInput(int nargs, char ** args, struct inputParameters* inpp){
    memset(inpp, 0, sizeof(struct inputParameters));
    
    if (nargs == 2 && (strcmp(args[1], "--help")==0 || strcmp(args[1], "-h"))) {
        printhelp();
        return 1;
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
            inpp->simulateTillStill = 1;
        if (strcmp(cmd, "-t")==0)
            inpp->time = atof(args[arg++]);
        if (strcmp(cmd, "r")==0)
            strcpy(inpp->reportfile,args[arg++]);
        if (strcmp(cmd, "--seed")==0)
            inpp->seed = atol(args[arg++]);
        if (strcmp(cmd, "-N")==0)
            inpp->width = inpp->height = atoll(args[arg++]);
    }
    return 0;
}

//Integer map opterations
int generateMapInt(int *map, double density, unsigned long seed, struct vector2u size) {
    unsigned int s = seed;

    size_t elems = size.x * size.y;
    size_t i, rest = elems-elems%4;
    
    //No multithead for consistancy reasons.
    for(i=0; i<elems; i+=4) {
            map[i+0] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
            map[i+1] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
            map[i+2] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
            map[i+3] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
    }
    for(i=rest; i<elems; i++) {
        map[i+0] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
    }
    
    return 0;
}

//Integer map opterations
int generateMapIntP(int **map, double density, unsigned long seed, struct vector2u size) {
    unsigned int s = seed;

    size_t elems = size.x * size.y;
    size_t x,y;
    
    //No multithead for consistancy reasons.
    for(x=0; x<size.x; x++)
        for(y=0; y<size.y; y++)
            map[x][y] = (int)(density>=((double)rand_r(&s)/(double)RAND_MAX));
    
    return 0;
}

size_t countAliveInt(int *map, struct vector2u size) {
    size_t elems = size.x * size.y;
    size_t i, acc1, acc2, acc3, acc4;
    acc1 = acc2 = acc3 = acc4 = 0;
    size_t rest = elems-(elems % 4);
    
    for(i=0; i<elems; i+=4) {
        acc1 += map[i+0];
        acc2 += map[i+1];
        acc3 += map[i+2];
        acc4 += map[i+3];
    }
    for(i=rest; i<elems; i++)
        acc1 += map[i];
    
    return acc1 + acc2 + acc3 + acc4;
}


size_t countAliveIntSSE(int *pDatapointer, struct vector2u size) {
    //Meh
}

//binary map operations
int generateMapBinary(void *pDatapointer, double density,unsigned long seed, struct vector2u size);
size_t countAliveBinary(void *pDatapointer, struct vector2u size);