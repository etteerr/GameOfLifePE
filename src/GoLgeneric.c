#include <string.h>
#include <x86intrin.h>
#include <stdio.h>

#include "GoLgeneric.h"

void printhelp() {
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

void printInputParameters(struct inputParameters * p) {
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

int parseInput(int nargs, char ** args, struct inputParameters* inpp) {
    memset(inpp, 0, sizeof (struct inputParameters));

    if (nargs == 2 && (strcmp(args[1], "--help") == 0 || strcmp(args[1], "-h"))) {
        printhelp();
        return 1;
    }

    //Extract arguments
    int arg = 1;

    while (arg < nargs) {
        char *cmd = args[arg++];

        if (strcmp(cmd, "-w") == 0)
            inpp->width = atoll(args[arg++]);
        if (strcmp(cmd, "-h") == 0)
            inpp->height = atoll(args[arg++]);
        if (strcmp(cmd, "-d") == 0)
            inpp->density = atof(args[arg++]);
        if (strcmp(cmd, "-s") == 0)
            inpp->steps = atoll(args[arg++]);
        if (strcmp(cmd, "-l") == 0)
            inpp->simulateTillStill = 1;
        if (strcmp(cmd, "-t") == 0)
            inpp->time = atof(args[arg++]);
        if (strcmp(cmd, "r") == 0)
            strcpy(inpp->reportfile, args[arg++]);
        if (strcmp(cmd, "--seed") == 0)
            inpp->seed = atol(args[arg++]);
        if (strcmp(cmd, "-N") == 0)
            inpp->width = inpp->height = atoll(args[arg++]);
    }
    return 0;
}

//Integer map opterations

int generateMapInt(int *map, double density, unsigned long seed, struct vector2u size) {
    unsigned int s = seed;

    size_t elems = size.x * size.y;
    size_t i, rest = elems % 4;

    //No multithead for consistancy reasons.
    if (elems >= 4)
        for (i = 0; i < elems - 3; i += 4) {
            map[i + 0] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));
            map[i + 1] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));
            map[i + 2] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));
            map[i + 3] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));
        }
    for (i = elems - rest; i < elems; i++) {
        map[i + 0] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));
    }

    return 0;
}

//Integer map opterations

int generateMapIntP(int **map, double density, unsigned long seed, struct vector2u size) {
    unsigned int s = seed;

    size_t elems = size.x * size.y;
    size_t x, y;

    //No multithead for consistancy reasons.
    for (x = 0; x < size.x; x++)
        for (y = 0; y < size.y; y++)
            map[x][y] = (int) (density >= ((double) rand_r(&s) / (double) RAND_MAX));

    return 0;
}

size_t countAliveInt(int *map, struct vector2u size) {
    size_t elems = size.x * size.y;
    size_t i, acc1, acc2, acc3, acc4;
    acc1 = acc2 = acc3 = acc4 = 0;
    size_t rest = (elems % 4);


    if (elems >= 4)
        for (i = 0; i < elems - 3; i += 4) {
            acc1 += map[i + 0];
            acc2 += map[i + 1];
            acc3 += map[i + 2];
            acc4 += map[i + 3];
        }
    for (i = elems - rest; i < elems; i++)
        acc1 += map[i];

    return acc1 + acc2 + acc3 + acc4;
}

size_t countAliveIntSSE(int *pDatapointer, struct vector2u size) {
    //Meh
    printf("countAliveIntSSE: Not implemented\n");
}

//binary map operations

int generateMapBinary(void *pDatapointer, double density, unsigned long seed, struct vector2u size) {
    size_t i, rest;
    size_t elems = size.x * size.y;
    uchar buffer = 0;
    unsigned int s = seed;

    rest = elems % 8;

    for (i = 0; i < elems / 8; i++) {
        for (int j = 7; j >= 0; j--)
            buffer |= ((uchar) (density >= ((double) rand_r(&s) / (double) RAND_MAX))) << j;

        *(uchar*) (pDatapointer + i) = buffer; //WARNING: Raw pointer arithmetics is in byte!
        buffer = 0;
    }

    //Set buffer to 0, now we are not going to update All
    buffer = 0;

    for (int j = 7; j >= 8 - rest; j--)
        buffer |= ((uchar) (density >= ((double) rand_r(&s) / (double) RAND_MAX))) << j;

    *(uchar*) (pDatapointer + i) = buffer; //WARNING: Raw pointer arithmetics is in byte!

    return 0;
}

size_t countAliveBinary(void *pDatapointer, struct vector2u size) {
    //Note: bit order does not matter, as long as the amount of cells are counted
    //Take care on the last bytes though...


    /*
        int __builtin_popcount (unsigned int)
        Generates the popcntl machine instruction. 
        int __builtin_popcountl (unsigned long)
        Generates the popcntl or popcntq machine instruction, depending on the size of unsigned long. 
        int __builtin_popcountll (unsigned long long)
        Generates the popcntq machine instruction.
     */

    size_t count = 0;
    size_t acc1, acc2, acc3, acc4;
    acc1 = acc2 = acc3 = acc4 = 0;
    size_t elems = size.x * size.y;

    size_t lllim = elems / (sizeof (unsigned long long)*8);

    size_t left = lllim % 4;
    if (lllim >= 4)
        for (size_t i = 0; i < (lllim - 3); i += 4) {
            acc1 += __builtin_popcountll(*((unsigned long long*) (pDatapointer) + i));
            acc2 += __builtin_popcountll(*((unsigned long long*) (pDatapointer) + i + 1));
            acc3 += __builtin_popcountll(*((unsigned long long*) (pDatapointer) + i + 2));
            acc4 += __builtin_popcountll(*((unsigned long long*) (pDatapointer) + i + 3));
        }

    for (size_t i = lllim - left; i < lllim; i++)
        acc1 += __builtin_popcountll(*((unsigned long long*) (pDatapointer) + i));

    left = elems % (sizeof (unsigned long long)*8); //left bits
    size_t leftbytes = left / 8 + ((int) (left % 8 != 0)); //left bytes + padded
    size_t totalBytes = elems / 8 + ((int) (elems % 8 != 0));
    for (size_t i = totalBytes - leftbytes; i < totalBytes; i++) {
        acc1 += __builtin_popcount(*((uchar*) (pDatapointer + i)));
    }

    return acc1 + acc2 + acc3 + acc4;
}