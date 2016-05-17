#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void** alloc_2d(size_t y_size, size_t x_size, size_t element_size)
{
    const size_t x_size_padded = (x_size + 128) & 0xFFFFFF80;

    uint8_t *data = (uint8_t*)calloc(x_size_padded * y_size, element_size); 
    void   **res  = (void**)  calloc(y_size,                 sizeof(void*));

    if (data == 0 || res == 0)
    {
        free (data);
        free (res);
        res = 0;
        goto exit;
    }

    for (size_t i = 0; i < y_size; ++i)
        res[i] = data + (i * x_size_padded * element_size);
 
exit:
    return res;
}
