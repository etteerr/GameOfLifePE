/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   defines.h
 * Author: Erwin Diepgrond <e.j.diepgrond@gmail.com>
 *
 * Created on May 19, 2017, 2:28 PM
 */

#ifndef DEFINES_H
#define DEFINES_H

//Game of life defines
//TODO: Calculate flops
#define FLOPS_GOL_INT(X,Y,S,T) ((double)(9*X*Y*S)/(double)T)
#define MOPS_GOL_INT(X,Y,S,T) ((double)(10*X*Y*S)/(double)T)

//Define units
#define BYTE 1.0
#define KBYTE 1.0e3
#define MBYTE 1.0e6
#define GBYTE 1.0e9

#define DAY 24*HOUR
#define HOUR 3600.0
#define MIN 60.0
#define SEC 1.0
#define MSEC 1.0e-3
#define USEC 1.0e-6
#define NSEC 1.0e-9

#define FLOPS 1.0
#define KFLOPS 1.0e3
#define MFLOPS 1.0e6
#define GFLOPS 1.0e9

//Define calculations
#define GET_MEMSPEED(SIZE,TIME) ((double)SIZE/(double)TIME)
#define GET_FLOPS(OPS,TIME) ((double)OPS/(double)TIME)


#endif /* DEFINES_H */

