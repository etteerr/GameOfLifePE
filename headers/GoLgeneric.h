/* 
 * File:   GoLgeneric.h
 * Author: Erwin Diepgrond <e.j.diepgrond@gmail.com>
 *
 * Created on May 15, 2017, 2:39 PM
 * 
 * Generic Game of Life functions
 */

#ifndef GOLGENERIC_H
#define GOLGENERIC_H

#include <stdlib.h>

#define get_rm(DATA,X,Y) DATA[y*width+x]
//#define get_cm(DATA,X,Y) DATA[y+height*x]

struct inputParameters {
    size_t width, height, steps;
    double density,time;
    int simulateTillStill;
    unsigned long seed;
    char reportfile[1024]; //To be on the save size, its only 1kb
};

struct vector2u {
    size_t x,y; //width (x) height (y)
};
struct report {
    char runName[64];
    struct vector2u size;
    size_t iteration;
    size_t aliveCount;
    double iterationTimeSecond;
    double totalRuntime;
    double totalIOtime;
};

#ifdef __cplusplus
extern "C" {
#endif
    
    
    //--------------Reporter functions---------------------//
    /**
     * Starts a reporter session
     *  Will write header to empty file
     * @param filename file to write to
     * @param append if >0, will append to file
     * @param buffer The file IO buffer. set to 1 Mb?
     * @return 0 on success
     */
    int report_initialize(char * filename, int append, size_t buffer);
    
    /**
     * Writes report line to report
     *  Note that this is expected every few or even every iteration.
     *  Each report is a single line in the CSV file.
     *  runName is important to identify the run case. (in analysis)
     * @param r pointer to a report
     */
    void report_write(struct report * r);
    
    
    //--------------Input and map functions (exIO)---------------------//
    /**
     * Parses main argument input and reports the inferred input parameters
     * @param nargs 
     * @param args
     * @param inpp A pointer to a valid memory address for struct inputParameters
     * @return 0 on success
     */
    int parseInput(int nargs, char ** args, struct inputParameters* inpp);
    /**
     * Prints help info for parseInput (User level) 
     */
    void printhelp();
    
    //Integer map opterations
    /**
     * Generates a map and stores this in the memory given by the pointer
     * @param pDatapointer
     * @param seed
     * @param size in vector2 unsigned
     * @return 0 on success
     */
    int generateMapInt(int *pDatapointer, double density, unsigned long seed, struct vector2u);
     /**
     * Generates a map and stores this in the memory given by the pointer
     * This version is int **
     * @param pDatapointer
     * @param seed
     * @param size in vector2 unsigned
     * @return 0 on success
     */
    int generateMapIntP(int **pDatapointer, double density, unsigned long seed, struct vector2u);
    /**
     * Counts alive members in a interger map
     * @param pDatapointer
     * @param size of map
     * @return alive count
     */
    size_t countAliveInt(int *pDatapointer, struct vector2u);
    /**
     * Counts alive members in a interger map
     *  Data must be 64 bit aligned!
     *  aligned_alloc(64/8, size);
     * @param pDatapointer
     * @param size of map
     * @return alive count
     */
    size_t countAliveAVX(int *pDatapointer, struct vector2u);
    /**
     * Counts alive members in a interger map
     *  Data must be 64 bit aligned!
     *  aligned_alloc(64/8, size);
     * @param pDatapointer
     * @param size of map
     * @return alive count
     * 
     * Not implemented
     */
    //size_t countAliveSSE(int *pDatapointer, struct vector2u);
    
    //binary map operations
    int generateMapBinary(void *pDatapointer, double density,unsigned long seed, struct vector2u);
    size_t countAliveBinary(void *pDatapointer, struct vector2u);


#ifdef __cplusplus
}
#endif

#endif /* GOLGENERIC_H */

