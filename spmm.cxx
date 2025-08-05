#include <iostream>
#include <stdint.h>
#include <string.h>
#include "simd_common.h"

void cpu_transpose(int *mat, int n_col, int n_row)
{
    int *tmp = (int*)malloc(sizeof(int) * n_col * n_row);

    for (int idx_y = 0; idx_y < n_row; ++idx_y)
    {
        for (int idx_x = 0; idx_x < n_col; ++idx_x)
        {
            *(tmp + idx_x*n_row + idx_y) = *(mat + idx_y*n_col + idx_x);
        }
    }

    memcpy(mat, tmp, n_col*n_row*sizeof(int));
    free(tmp);
}

void cpu_transpose(float *mat, int n_col, int n_row)
{
    float *tmp = (float*)malloc(sizeof(float) * n_col * n_row);

    for (int idx_y = 0; idx_y < n_row; ++idx_y)
    {
        for (int idx_x = 0; idx_x < n_col; ++idx_x)
        {
            *(tmp + idx_x*n_row + idx_y) = *(mat + idx_y*n_col + idx_x);
        }
    }

    memcpy(mat, tmp, n_col*n_row*sizeof(float));
    free(tmp);
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
                    //idx_mat[(idx_outer_y + idx_inner_y++)*(sz/l/16) + idx_out_x/16] |= inner_y << ((idx_out_x % 16)*2);
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

void pack_mat(int *mat, int n_col, int n_row)
{
    for (int idx_y = 0; idx_y < n_row; ++idx_y)
    {
        for (int x_outer = 0; x_outer < n_col; x_outer += 16)
        {
            int *src_cell = mat + idx_y*n_col + x_outer;
            int *dst_cell = mat + idx_y*(n_col/16) + x_outer/16;

            *dst_cell = 0;
            for (int x_inner = 0; x_inner < 16; ++x_inner)
            {
                *dst_cell |= *(src_cell + x_inner) << (x_inner * 2);
            }
        }
    }
}
