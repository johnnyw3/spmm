#include <iostream>
#include <stdint.h>
#include <string.h>
#include "simd_common.h"

void cpu_transpose(float *mat, int n_col, int n_row)
{
    float tmp[n_col*n_row]; 

    for (int idx_y = 0; idx_y < n_row; ++idx_y)
    {
        for (int idx_x = 0; idx_x < n_col; ++idx_x)
        {
            *(tmp + idx_x*n_row + idx_y) = *(mat + idx_y*n_col + idx_x);
        }
    }

    memcpy(mat, tmp, n_col*n_row*sizeof(float));
}

void cpu_transpose(float *mat, int n)
{
    for (int idx_y = 0; idx_y < n; ++idx_y)
    {
        for (int idx_x = idx_y+1; idx_x < n; ++idx_x)
        {
            float temp_upper = *(mat + idx_y*n + idx_x);
            *(mat + idx_y*n + idx_x) = *(mat + idx_x*n + idx_y);
            *(mat + idx_x*n + idx_y) = temp_upper;
        }
    }
}

void squash_matrix(float *in_mat, float *out_mat, int *idx_mat, 
                   int n, int m, int l, int sz)
{
    int idx_out_x = 0, idx_outer_y = 0, idx_inner_y = 0;
    for (int outer_y = 0; outer_y < sz; outer_y += m)
    {
        for (int outer_x = 0; outer_x < sz; outer_x += l)
        {
            int idx_inner_y = 0;
            for (int inner_y = 0; inner_y < m; ++inner_y)
            {
                if (in_mat[(outer_y + inner_y)*sz + outer_x])
                {
                    memcpy(out_mat + (idx_outer_y + idx_inner_y)*sz + outer_x, in_mat + (outer_y + inner_y)*sz + outer_x, l*sizeof(float)); 
                    idx_mat[(idx_outer_y + idx_inner_y++)*(sz/l) + idx_out_x] = inner_y;
                }
            }
            idx_out_x++;
        }
        idx_outer_y += n;
        idx_inner_y  = 0;
        idx_out_x    = 0;
    }
}
