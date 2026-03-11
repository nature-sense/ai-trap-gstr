// Single translation unit that compiles the stb_image_write implementation.
// No other .cpp file should define STB_IMAGE_WRITE_IMPLEMENTATION.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_ASSERT(x)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "stb_image_write.h"
#pragma GCC diagnostic pop
