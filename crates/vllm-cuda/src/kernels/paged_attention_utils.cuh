#ifndef PAGED_ATTENTION_UTILS_CUH
#define PAGED_ATTENTION_UTILS_CUH

#include <cuda_runtime.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

#define PARTITION_SIZE 256

#endif // PAGED_ATTENTION_UTILS_CUH
