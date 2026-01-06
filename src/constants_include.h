/**
 * This header file provides a switchable mechanism for using either
 * runtime-loaded constants or compile-time constants
 */

#ifndef CONSTANTS_INCLUDE_H
#define CONSTANTS_INCLUDE_H

#ifdef USE_EXTENDED_PARAMETERS

// for extended d3 system parameters,
// 103 elements are supported, ech have 7 reference points for C6 computation

#define NUM_ELEMENTS 103
#define NUM_REF_C6 7
#define NUM_C6AB_ENTRIES 3 

#include "gen_constants_extended.h"

#else

// for standard d3 system parameters,
// 95 elements are supported, each have 5 reference points for C6 computation

#define NUM_ELEMENTS 95
#define NUM_REF_C6 5
#define NUM_C6AB_ENTRIES 3


#include "gen_constants_original.h"

#endif // USE_EXTENDED_PARAMETERS

#endif  // CONSTANTS_INCLUDE_H
