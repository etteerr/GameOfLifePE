#include <stdio.h> 


/***********************

Conway Game of Life

 ************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "GoLgeneric.h"
#include "defines.h"

int bwidth, bheight, nsteps;
int i, j, n, im, ip, jm, jp, ni, nj, nsum, isum;
int **old, **new;
float x;
struct timeval start;
struct timeval end;
double rtime;

// update board for step n

void doTimeStep(int n) {
    /* corner boundary conditions */
    old[0][0] = old[bwidth][bheight];
    old[0][bheight + 1] = old[bwidth][1];
    old[bwidth + 1][bheight + 1] = old[1][1];
    old[bwidth + 1][0] = old[1][bheight];

    /* left-right boundary conditions */
    for (i = 1; i <= bwidth; i++) {
        old[i][0] = old[i][bheight];
        old[i][bheight + 1] = old[i][1];
    }

    /* top-bottom boundary conditions */
    for (j = 1; j <= bheight; j++) {
        old[0][j] = old[bwidth][j];
        old[bwidth + 1][j] = old[1][j];
    }

    // update board
    for (i = 1; i <= bwidth; i++) {
        for (j = 1; j <= bheight; j++) {
            im = i - 1;
            ip = i + 1;
            jm = j - 1;
            jp = j + 1;

            nsum = old[im][jp] + old[i][jp] + old[ip][jp]
                    + old[im][j] + old[ip][j]
                    + old[im][jm] + old[i][jm] + old[ip][jm];

            switch (nsum) {
                    // a new organism is born
                case 3:
                    new[i][j] = 1;
                    break;
                    // nothing happens
                case 2:
                    new[i][j] = old[i][j];
                    break;
                    // the oranism, if any, dies
                default:
                    new[i][j] = 0;
            }
        }
    }

    /* copy new state into old state */
    for (i = 1; i <= bwidth; i++) {
        for (j = 1; j <= bheight; j++) {
            old[i][j] = new[i][j];
        }
    }
}

int main(int argc, char *argv[]) {
    
    /* Get Parameters */
    struct inputParameters p;
    parseInput(argc, argv, &p);
    printInputParameters(&p);
    bwidth = p.width;
    bheight = p.height;
    nsteps = p.steps;

    /* allocate arrays */
    ni = bwidth + 2; /* add 2 for left and right ghost cells */
    nj = bheight + 2;
    printf("Allocating %i bytes\n"
            "That's %f Mb.\n",
            nj * sizeof (int*)*2 + nj * sizeof (int)*ni * 2,
            (double) (nj * sizeof (int*)*2 + nj * sizeof (int)*ni * 2) / 1.0e6);
    old = malloc(ni * sizeof (int*));
    new = malloc(ni * sizeof (int*));

    for (i = 0; i < ni; i++) {
        old[i] = malloc(nj * sizeof (int));
        new[i] = malloc(nj * sizeof (int));
    }

    /*  initialize board */
    struct vector2u size;
    size.x = p.width;
    size.y = p.height;
    generateMapIntP(old, p.density, p.seed, size);
    //Count initial
    isum = 0;
    for (i = 1; i <= bwidth; i++) {
        for (j = 1; j <= bheight; j++) {
            isum = isum + old[i][j];
        }
    }
    
    printf("Number of live cells = %d\n", isum);

    if (gettimeofday(&start, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    /*  time steps */
    for (n = 0; n < nsteps; n++) {
        doTimeStep(n);
    }

    if (gettimeofday(&end, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    // compute running time
    rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) -
            (start.tv_sec + (start.tv_usec / 1000000.0));

    /*  Iterations are done; sum the number of live cells */
    isum = 0;
    for (i = 1; i <= bwidth; i++) {
        for (j = 1; j <= bheight; j++) {
            isum = isum + new[i][j];
        }
    }

    printf("Number of live cells = %d\n", isum);
    fprintf(stderr, "Game of Life took %10.3f seconds\n", rtime);
    printf("Game of Life did %e FLOPS\n", FLOPS_GOL_INT(bwidth, bheight, nsteps, rtime));
    printf("Processing %f Gbyte of info per second\n", (MOPS_GOL_INT(bwidth, bheight, nsteps, rtime)*4)/GBYTE);

    return 0;
}
