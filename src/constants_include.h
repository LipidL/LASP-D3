/**
 * This header file provides a switchable mechanism for using either
 * runtime-loaded constants or compile-time constants
 */

#ifndef CONSTANTS_INCLUDE_H
#define CONSTANTS_INCLUDE_H

// If BUILD_WITH_STATIC_CONSTANTS is defined, use the generated constants header
// Otherwise use the original runtime-loading approach
#ifdef BUILD_WITH_STATIC_CONSTANTS
#include "gen_constants.h"
#else
#include "constants.h"
#endif

#endif // CONSTANTS_INCLUDE_H
