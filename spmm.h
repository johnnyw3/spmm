#ifndef __GEMM_H__
#define __GEMM_H__ 1
#include <x86intrin.h>
#include <pthread.h>
#include <string.h>

#define US_PER_S 1000000
#define GIGA     1000000000

void simd_spmm(float * __restrict mat1, float * __restrict mat2, float * __restrict dst, int sz, int n, int m, int l);
void squash_matrix(float *in_mat, float *out_mat, int *idx_mat, 
                   int n, int m, int l, int sz);
void* simd_spmm_worker(void *argv);
void* simd_spmm_worker_avx512(void *argv);
void cpu_transpose(float *mat, int n_col, int n_row);
void cpu_transpose(float *mat, int n);

typedef struct{
    float *mat1;
    float *mat2;
    float *dst;
    int    sz;
    int    n;
    int    m;
    int    l;
    int    th_id;
} spmmOptns;

void simd_spmm(float * __restrict mat1, float * __restrict mat2, float * __restrict dst, int sz, int n, int m, int l)
{
    spmmOptns thd_optns[NUM_THREADS];
    pthread_t thds[NUM_THREADS];

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        spmmOptns optn =  {mat1, mat2, dst, sz, n, m, l, th_id};
        thd_optns[th_id] = optn;
#ifdef USE_AVX512
        pthread_create(&thds[th_id], NULL, &simd_spmm_worker_avx512, (void*)&thd_optns[th_id]);
#else 
        pthread_create(&thds[th_id], NULL, &simd_spmm_worker, (void*)&thd_optns[th_id]);
#endif 
    }

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        pthread_join(thds[th_id], NULL);
    }

}

void* simd_spmm_worker(void *argv)
{
    spmmOptns *optns = (spmmOptns*)argv;
    float * const mat1 = optns->mat1;
    float * const mat2 = optns->mat2;
    float * const dst  = optns->dst;
    const int    sz   = optns->sz;
    const int    n    = optns->n;
    const int    m    = optns->m;
    const int    l    = optns->l;
    const int    th_id = optns->th_id;
    const int    thd_loop_sz = n / NUM_THREADS;
    const int    start_idx = thd_loop_sz * th_id;
    const int    stop_idx  = start_idx + thd_loop_sz;

    const int compression_ratio = m / n;
    constexpr int simd_ele_width  = SIMD_WIDTH  / sizeof(float);
    constexpr int block_ele_i = BLOCK_I / sizeof(float);
    constexpr int block_ele_j = BLOCK_J / sizeof(float);
    const int block_ele_k = BLOCK_K / sizeof(float) * compression_ratio;
    constexpr int sblock_ele_i = SBLOCK_I / sizeof(float);
    constexpr int sblock_ele_j = SBLOCK_J / sizeof(float);
    const int sblock_ele_k = SBLOCK_K / sizeof(float) * compression_ratio;
    //int vec_n = n / simd_ele_width;
    constexpr int block_ni = sblock_ele_i/block_ele_i;
    constexpr int block_nj = sblock_ele_j/block_ele_j;
    const int block_nk = sblock_ele_k/block_ele_k;

    // STUB

    return NULL;
}

#ifdef USE_AVX512
/*
 * This just has my GEMM code copied in to use as a starting place. It will be 
 * modified to work with SpMM in future ... */
void* simd_spmm_worker_avx512(void *argv)
{
    spmmOptns *optns = (spmmOptns*)argv;
    float * const mat1 = optns->mat1;
    float * const mat2 = optns->mat2;
    float * const dst  = optns->dst;
    const int    sz   = optns->sz;
    const int    n    = optns->n;
    const int    m    = optns->m;
    const int    l    = optns->l;
    const int    th_id = optns->th_id;
    const int    thd_loop_sz = n / NUM_THREADS;
    const int    start_idx = thd_loop_sz * th_id;
    const int    stop_idx  = start_idx + thd_loop_sz;

    const int compression_ratio = m / n;
    constexpr int simd_ele_width  = SIMD_WIDTH  / sizeof(float);
    constexpr int block_ele_i = BLOCK_I / sizeof(float);
    constexpr int block_ele_j = BLOCK_J / sizeof(float);
    const int block_ele_k = BLOCK_K / sizeof(float) * compression_ratio;
    constexpr int sblock_ele_i = SBLOCK_I / sizeof(float);
    constexpr int sblock_ele_j = SBLOCK_J / sizeof(float);
    const int sblock_ele_k = SBLOCK_K / sizeof(float) * compression_ratio;
    //int vec_n = n / simd_ele_width;
    constexpr int block_ni = sblock_ele_i/block_ele_i;
    constexpr int block_nj = sblock_ele_j/block_ele_j;
    const int block_nk = sblock_ele_k/block_ele_k;

    float * __restrict mat1_ptr, * __restrict mat2_ptr, * __restrict dst_ptr;
    float * __restrict mat1_ptr2, * __restrict dst_ptr2;
    float * __restrict mat2_ptr2, * __restrict mat2_ptr3, * __restrict mat2_ptr4;

    for (int i_outer = start_idx; i_outer < stop_idx; i_outer += sblock_ele_i)
    {
        for (int j_outer = 0; j_outer < n; j_outer += sblock_ele_j)
        {
            for (int k_outer = 0; k_outer < n; k_outer += sblock_ele_k)
            {
                float packed_b[sblock_ele_j*sblock_ele_k] __attribute__ ((__aligned__(64)));
                float packed_a[sblock_ele_k*sblock_ele_i] __attribute__ ((__aligned__(64)));

                mat1_ptr = mat1 + (i_outer)*n + k_outer;
                mat2_ptr = packed_a;
                for (int idx = 0; idx < sblock_ele_i;)
                {
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    memcpy(mat2_ptr, mat1_ptr + jdx, BLOCK_K);
                    mat2_ptr += block_ele_k*block_ele_i;
                    }
                    mat1_ptr += n;

                    ++idx;
                    if (! (idx%block_ele_i) )
                        mat2_ptr -= block_ele_k*block_ele_i - block_ele_k;
                    else
                        mat2_ptr -= block_ele_k*(block_ele_i)*block_nk - block_ele_k;
                }

                mat2_ptr = mat2 + (j_outer)*n + k_outer;
                mat1_ptr = packed_b;

                for (int idx = 0; idx < sblock_ele_j;)
                {
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    memcpy(mat1_ptr, mat2_ptr + jdx, BLOCK_K);
                    mat1_ptr += block_ele_k*block_ele_j;
                    }
                    mat2_ptr += n;
                    ++idx;

                    if (! (idx%block_ele_j) )
                        mat1_ptr -= block_ele_k*block_ele_j - block_ele_k;
                    else
                        mat1_ptr -= block_ele_k*(block_ele_j)*block_nk - block_ele_k;
                }

    for (int i_outer2 = 0; i_outer2< sblock_ele_i; i_outer2+= block_ele_i)
    {
        for (int j_outer2 = 0; j_outer2< sblock_ele_j; j_outer2 += block_ele_j)
        {
            for (int k_outer2 = 0; k_outer2< sblock_ele_k; k_outer2 += block_ele_k)
            {
                for (int i_inner = 0; i_inner < block_ele_i; i_inner += 2)
                {
                    mat1_ptr = packed_a + (i_outer2*block_nk + i_inner)*block_ele_k + k_outer2*block_ele_i;
                    mat1_ptr2 = mat1_ptr + block_ele_k;

                    dst_ptr = dst + (i_outer + i_outer2 + i_inner)*n + j_outer + j_outer2;
                    dst_ptr2 = dst_ptr + n;
                    
                    for (int j_inner = 0; j_inner < block_ele_j; j_inner += simd_ele_width)
                    {

                            __m256 a_vec, a_vec2, b_vec, b_vec2, b_vec3, b_vec4;
                            __m256 dst2, dst3; // = _mm256_setzero_ps();

                            __m256 sums0 = _mm256_setzero_ps();
                            __m256 sums1 = _mm256_setzero_ps();
                            __m256 sums2 = _mm256_setzero_ps();
                            __m256 sums3 = _mm256_setzero_ps();
                            __m256 sums4 = _mm256_setzero_ps();
                            __m256 sums5 = _mm256_setzero_ps();
                            __m256 sums6 = _mm256_setzero_ps();
                            __m256 sums7 = _mm256_setzero_ps();
                            __m256 sums20 = _mm256_setzero_ps();
                            __m256 sums21 = _mm256_setzero_ps();
                            __m256 sums22 = _mm256_setzero_ps();
                            __m256 sums23 = _mm256_setzero_ps();
                            __m256 sums24 = _mm256_setzero_ps();
                            __m256 sums25 = _mm256_setzero_ps();
                            __m256 sums26 = _mm256_setzero_ps();
                            __m256 sums27 = _mm256_setzero_ps();

                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 0)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;

                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm256_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm256_load_ps( mat2_ptr3+ k_inner );
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );
                                sums0 = _mm256_fmadd_ps(a_vec, b_vec, sums0);
                                sums20 = _mm256_fmadd_ps(a_vec2, b_vec, sums20);

                                sums1 = _mm256_fmadd_ps(a_vec, b_vec2, sums1);
                                sums21 = _mm256_fmadd_ps(a_vec2, b_vec2, sums21);

                                sums2 = _mm256_fmadd_ps(a_vec, b_vec3, sums2);
                                sums22 = _mm256_fmadd_ps(a_vec2, b_vec3, sums22);

                                sums3 = _mm256_fmadd_ps(a_vec, b_vec4, sums3);
                                sums23 = _mm256_fmadd_ps(a_vec2, b_vec4, sums23);
                            }

                            __m256 const upper = _mm256_unpacklo_ps(sums0, sums2); // 0100 1110
                            __m256 const lower = _mm256_unpackhi_ps(sums0, sums2);
                            __m256 const upper2 = _mm256_unpacklo_ps(sums1, sums3); // 0100 1110
                            __m256 const lower2 = _mm256_unpackhi_ps(sums1, sums3);
                            __m256 const upper1 = _mm256_unpacklo_ps(sums20, sums22); // 0100 1110
                            __m256 const lower1 = _mm256_unpackhi_ps(sums20, sums22);
                            __m256 const upper12 = _mm256_unpacklo_ps(sums21, sums23); // 0100 1110
                            __m256 const lower12 = _mm256_unpackhi_ps(sums21, sums23);
                            __m256 const res4 = _mm256_add_ps(lower, upper);
                            __m256 const res5 = _mm256_add_ps(lower2, upper2);
                            __m256 const res14= _mm256_add_ps(lower1, upper1);
                            __m256 const res15= _mm256_add_ps(lower12, upper12);

                            __m256 const upper3 = _mm256_unpacklo_ps(res4, res5); 
                            __m256 const lower3 = _mm256_unpackhi_ps(res4, res5); 
                            __m256 const upper13 = _mm256_unpacklo_ps(res14, res15); 
                            __m256 const lower13 = _mm256_unpackhi_ps(res14, res15); 
                            dst2 = _mm256_add_ps(lower3, upper3);
                            dst3 = _mm256_add_ps(lower13, upper13);


                            mat2_ptr  = mat2_ptr + 4*block_ele_k;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm256_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + k_inner );
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );

                                sums4 = _mm256_fmadd_ps(a_vec, b_vec, sums4);
                                sums24 = _mm256_fmadd_ps(a_vec2, b_vec, sums24);

                                sums5 = _mm256_fmadd_ps(a_vec, b_vec2, sums5);
                                sums25 = _mm256_fmadd_ps(a_vec2, b_vec2, sums25);

                                sums6 = _mm256_fmadd_ps(a_vec, b_vec3, sums6);
                                sums26 = _mm256_fmadd_ps(a_vec2, b_vec3, sums26);

                                sums7 = _mm256_fmadd_ps(a_vec, b_vec4, sums7);
                                sums27 = _mm256_fmadd_ps(a_vec2, b_vec4, sums27);
                            }

                            __m256 const upper4 = _mm256_unpacklo_ps(sums4, sums6); // 0100 1110
                            __m256 const lower4 = _mm256_unpackhi_ps(sums4, sums6);
                            __m256 const upper5 = _mm256_unpacklo_ps(sums5, sums7); // 0100 1110
                            __m256 const lower5 = _mm256_unpackhi_ps(sums5, sums7);
                            __m256 const upper14 = _mm256_unpacklo_ps(sums24, sums26); // 0100 1110
                            __m256 const lower14 = _mm256_unpackhi_ps(sums24, sums26);
                            __m256 const upper15 = _mm256_unpacklo_ps(sums25, sums27); // 0100 1110
                            __m256 const lower15 = _mm256_unpackhi_ps(sums25, sums27);
                            __m256 const res1 = _mm256_add_ps(lower4, upper4);
                            __m256 const res2 = _mm256_add_ps(lower5, upper5);
                            __m256 const res11 = _mm256_add_ps(lower14, upper14);
                            __m256 const res12 = _mm256_add_ps(lower15, upper15);


                        __m256 const upper18 = _mm256_load_ps(dst_ptr2);
                        __m256 const upper8 = _mm256_load_ps(dst_ptr);
                            __m256 const upper6 =  _mm256_unpacklo_ps(res1, res2); 
                            __m256 const lower6 = _mm256_unpackhi_ps(res1, res2); 
                            __m256 const upper16 = _mm256_unpacklo_ps(res11, res12); 
                            __m256 const lower16 = _mm256_unpackhi_ps(res11, res12); 
                            __m256 const res3 = _mm256_add_ps(lower6, upper6);
                            __m256 const res13= _mm256_add_ps(lower16, upper16);


                            __m256 const lower17 = _mm256_permute2f128_ps(dst3, res13, 0x21 ); 
                            __m256 const lower7 = _mm256_permute2f128_ps(dst2, res3 , 0x21); 
                            __m256 const upper17 = _mm256_blend_ps(dst3, res13, 0xF0); 
                            __m256 const upper7 = _mm256_blend_ps(dst2, res3, 0xF0); 
                            dst2 = _mm256_add_ps(lower7, upper7);
                            dst3 = _mm256_add_ps(lower17, upper17);

                        //__m256 sums2 = _mm256_load_ps(temp); 
                        //__m256 sums2 = _mm256_setzero_ps(); 
                        dst2 = _mm256_add_ps(dst2, upper8);
                        dst3 = _mm256_add_ps(dst3, upper18);
                        _mm256_store_ps(dst_ptr, dst2);
                        _mm256_store_ps(dst_ptr2, dst3);
                        dst_ptr += simd_ele_width;
                        dst_ptr2 += simd_ele_width;

                       }}} 
                    }
                }
            }
        }
    }
    return NULL;
}
#endif // USE_AVX512

#endif  // __GEMM_H__
