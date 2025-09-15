/**
 * This header file provides a switchable mechanism for using either
 * runtime-loaded constants or compile-time constants
 */

#ifndef CONSTANTS_INCLUDE_H
#define CONSTANTS_INCLUDE_H

// d3 system parameters
#define NUM_ELEMENTS 103
#define NUM_REF_C6 7    // each element have at most 7 reference points for C6 computation
#define NUM_C6AB_ENTRIES 3  // first entry: C6; second entry: CN of first element; third entry: CN of second element


#include "gen_constants.h"


#endif  // CONSTANTS_INCLUDE_H
