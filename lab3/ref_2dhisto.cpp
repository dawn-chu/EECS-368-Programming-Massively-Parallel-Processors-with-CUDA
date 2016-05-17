#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ref_2dhisto.h"

int ref_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{

    // Zero out all the bins
    memset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(bins[0]));

    for (size_t j = 0; j < height; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            const uint32_t value = input[j][i];

            uint8_t *p = (uint8_t*)bins;

            // Increment the appropriate bin, but do not roll-over the max value
            if (p[value] < UINT8_MAX)
                ++p[value];
        }
    }

    return 0;
}
