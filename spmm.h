#ifndef __GEMM_H__
#define __GEMM_H__ 1
#include <x86intrin.h>
#include <pthread.h>
#include <string.h>

#define US_PER_S 1000000
#define GIGA     1000000000

void simd_spmm(float * __restrict mat1, float * __restrict mat2, int * __restrict mat2i, float * __restrict dst, int sz, int n, int m, int l);
void squash_matrix(float *in_mat, float *out_mat, int *idx_mat, 
                   int n, int m, int l, int sz);
void* simd_spmm_worker(void *argv);
void* simd_spmm_worker_avx512(void *argv);
void cpu_transpose(int   *mat, int n_col, int n_row);
void cpu_transpose(float *mat, int n_col, int n_row);
void cpu_transpose(float *mat, int n);

typedef struct{
    float *mat1;
    float *mat2;
    int   *mat2i;
    float *dst;
    int    sz;
    int    n;
    int    m;
    int    l;
    int    th_id;
} spmmOptns;

void simd_spmm(float * __restrict mat1, float * __restrict mat2, int * __restrict mat2i, float * __restrict dst, int sz, int n, int m, int l)
{
    spmmOptns thd_optns[NUM_THREADS];
    pthread_t thds[NUM_THREADS];

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        spmmOptns optn =  {mat1, mat2, mat2i, dst, sz, n, m, l, th_id};
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
    int   * const mat2i= optns->mat2i;
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
    const int compressed_sz = sz / compression_ratio;
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
    int   * __restrict mat2i_ptr;

    mat1_ptr = mat1;
    mat2_ptr = mat2;
    mat2i_ptr = mat2i;

    for (int i = 0; i < sz; i += 2)
    {
        mat1_ptr = mat1 + i*sz;
        mat1_ptr2= mat1_ptr + sz;

        dst_ptr = dst + i*sz;
        dst_ptr2 = dst_ptr + sz;

        for (int j = 0; j < sz; j += simd_ele_width)
        {
            __m512 sums0 = _mm512_setzero_ps();
            __m512 sums1 = _mm512_setzero_ps();
            __m512 sums2 = _mm512_setzero_ps();
            __m512 sums3 = _mm512_setzero_ps();
            __m512 sums4 = _mm512_setzero_ps();
            __m512 sums5 = _mm512_setzero_ps();
            __m512 sums6 = _mm512_setzero_ps();
            __m512 sums7 = _mm512_setzero_ps();
            __m512 sums8 = _mm512_setzero_ps();
            __m512 sums9 = _mm512_setzero_ps();
            __m512 sumsA = _mm512_setzero_ps();
            __m512 sumsB = _mm512_setzero_ps();
            __m512 sumsC = _mm512_setzero_ps();
            __m512 sumsD = _mm512_setzero_ps();
            __m512 sumsE = _mm512_setzero_ps();
            __m512 sumsF = _mm512_setzero_ps();
            __m512 sums20 = _mm512_setzero_ps();
            __m512 sums21 = _mm512_setzero_ps();
            __m512 sums22 = _mm512_setzero_ps();
            __m512 sums23 = _mm512_setzero_ps();
            __m512 sums24 = _mm512_setzero_ps();
            __m512 sums25 = _mm512_setzero_ps();
            __m512 sums26 = _mm512_setzero_ps();
            __m512 sums27 = _mm512_setzero_ps();
            __m512 sums28 = _mm512_setzero_ps();
            __m512 sums29 = _mm512_setzero_ps();
            __m512 sums2A = _mm512_setzero_ps();
            __m512 sums2B = _mm512_setzero_ps();
            __m512 sums2C = _mm512_setzero_ps();
            __m512 sums2D = _mm512_setzero_ps();
            __m512 sums2E = _mm512_setzero_ps();
            __m512 sums2F = _mm512_setzero_ps();

            mat2_ptr = mat2 + j*(compressed_sz);
            mat2_ptr2 = mat2_ptr + compressed_sz;
            mat2_ptr3 = mat2_ptr2 + compressed_sz;
            mat2_ptr4 = mat2_ptr3 + compressed_sz;

            mat2i_ptr = mat2i + (j/l)*(compressed_sz);

            __m512 b_vec, b_vec2, b_vec3, b_vec4, a_vec_t11, a_vec_t12, a_vec_t21,
                   a_vec_t22, a_vec, a_vec2, a_vec3, a_vec4, dst2, dst3;

            for (int k = 0; k < sz / compression_ratio; k += simd_ele_width)
            {
                __m512i idx_vec = _mm512_load_ps(mat2i_ptr + k);
                // TODO: this is for a fixed 2:4 4-wide vectorwide N:M sparsity.
                //       Should generate masks at runtime for parameterization.
                const __m512i move_mask = _mm512_set_epi32(0x1C, 0x1C, 0x18, 0x18, 0x14, 0x14, 0x10, 0x10, 0x0C, 0x0C, 0x08, 0x08, 0x04, 0x04, 0x00, 0x00);
                idx_vec = _mm512_add_epi32(idx_vec, move_mask);

                b_vec = _mm512_load_ps(mat2_ptr + k);
                b_vec2= _mm512_load_ps(mat2_ptr2+ k);
                b_vec3= _mm512_load_ps(mat2_ptr3+ k);
                b_vec4= _mm512_load_ps(mat2_ptr4+ k);

                a_vec_t11 = _mm512_load_ps(mat1_ptr + k*compression_ratio);
                a_vec_t12 = _mm512_load_ps(mat1_ptr + k*compression_ratio + simd_ele_width);
                a_vec_t21 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio);
                a_vec_t22 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio + simd_ele_width);
                a_vec = _mm512_permutex2var_ps(a_vec_t11, idx_vec, a_vec_t12);
                a_vec2= _mm512_permutex2var_ps(a_vec_t21, idx_vec, a_vec_t22);

                sums0 = _mm512_fmadd_ps(a_vec, b_vec, sums0);
                sums20 = _mm512_fmadd_ps(a_vec2, b_vec, sums20);

                sums1 = _mm512_fmadd_ps(a_vec, b_vec2, sums1);
                sums21 = _mm512_fmadd_ps(a_vec2, b_vec2, sums21);

                sums2 = _mm512_fmadd_ps(a_vec, b_vec3, sums2);
                sums22 = _mm512_fmadd_ps(a_vec2, b_vec3, sums22);

                sums3 = _mm512_fmadd_ps(a_vec, b_vec4, sums3);
                sums23 = _mm512_fmadd_ps(a_vec2, b_vec4, sums23);
            }

            __m512 const upper = _mm512_unpacklo_ps(sums0, sums2); 
            __m512 const lower = _mm512_unpackhi_ps(sums0, sums2);
            __m512 const upper2 = _mm512_unpacklo_ps(sums1, sums3);
            __m512 const lower2 = _mm512_unpackhi_ps(sums1, sums3);
            __m512 const upper1 = _mm512_unpacklo_ps(sums20, sums22);
            __m512 const lower1 = _mm512_unpackhi_ps(sums20, sums22);
            __m512 const upper12 = _mm512_unpacklo_ps(sums21, sums23);
            __m512 const lower12 = _mm512_unpackhi_ps(sums21, sums23);
            __m512 const res000 = _mm512_add_ps(lower, upper);
            __m512 const res001 = _mm512_add_ps(lower2, upper2);
            __m512 const res100 = _mm512_add_ps(lower1, upper1);
            __m512 const res101 = _mm512_add_ps(lower12, upper12);

            __m512 const upper3 = _mm512_unpacklo_ps(res000, res001);
            __m512 const lower3 = _mm512_unpackhi_ps(res000, res001);
            __m512 const upper13 = _mm512_unpacklo_ps(res100, res101);
            __m512 const lower13 = _mm512_unpackhi_ps(res100, res101);
            __m512 const res010 = _mm512_add_ps(lower3, upper3);
            __m512 const res110 = _mm512_add_ps(lower13, upper13);

            mat2_ptr += 8*compressed_sz;
            mat2_ptr2 = mat2_ptr + compressed_sz;
            mat2_ptr3 = mat2_ptr2 + compressed_sz;
            mat2_ptr4 = mat2_ptr3 + compressed_sz;
            mat2i_ptr += (8/l)*compressed_sz;

            for (int k = 0; k < sz / compression_ratio; k += simd_ele_width)
            {
                __m512i idx_vec = _mm512_load_ps(mat2i_ptr + k);
                // TODO: this is for a fixed 2:4 4-wide vectorwide N:M sparsity.
                //       Should generate masks at runtime for parameterization.
                const __m512i move_mask = _mm512_set_epi32(0x1C, 0x1C, 0x18, 0x18, 0x14, 0x14, 0x10, 0x10, 0x0C, 0x0C, 0x08, 0x08, 0x04, 0x04, 0x00, 0x00);
                idx_vec = _mm512_add_epi32(idx_vec, move_mask);

                b_vec = _mm512_load_ps(mat2_ptr + k);
                b_vec2= _mm512_load_ps(mat2_ptr2+ k);
                b_vec3= _mm512_load_ps(mat2_ptr3+ k);
                b_vec4= _mm512_load_ps(mat2_ptr4+ k);

                a_vec_t11 = _mm512_load_ps(mat1_ptr + k*compression_ratio);
                a_vec_t12 = _mm512_load_ps(mat1_ptr + k*compression_ratio + simd_ele_width);
                a_vec_t21 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio);
                a_vec_t22 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio + simd_ele_width);
                a_vec = _mm512_permutex2var_ps(a_vec_t11, idx_vec, a_vec_t12);
                a_vec2= _mm512_permutex2var_ps(a_vec_t21, idx_vec, a_vec_t22);

                sums8 = _mm512_fmadd_ps(a_vec, b_vec, sums8);
                sums28 = _mm512_fmadd_ps(a_vec2, b_vec, sums28);

                sums9 = _mm512_fmadd_ps(a_vec, b_vec2, sums9);
                sums29 = _mm512_fmadd_ps(a_vec2, b_vec2, sums29);

                sumsA = _mm512_fmadd_ps(a_vec, b_vec3, sumsA);
                sums2A = _mm512_fmadd_ps(a_vec2, b_vec3, sums2A);

                sumsB = _mm512_fmadd_ps(a_vec, b_vec4, sumsB);
                sums2B = _mm512_fmadd_ps(a_vec2, b_vec4, sums2B);
            }

            __m512 const upper4 = _mm512_unpacklo_ps(sums8, sumsA);
            __m512 const lower4 = _mm512_unpackhi_ps(sums8, sumsA);
            __m512 const upper5 = _mm512_unpacklo_ps(sums9, sumsB);
            __m512 const lower5 = _mm512_unpackhi_ps(sums9, sumsB);
            __m512 const upper14 = _mm512_unpacklo_ps(sums28, sums2A);
            __m512 const lower14 = _mm512_unpackhi_ps(sums28, sums2A);
            __m512 const upper15 = _mm512_unpacklo_ps(sums29, sums2B);
            __m512 const lower15 = _mm512_unpackhi_ps(sums29, sums2B);
            __m512 const res002 = _mm512_add_ps(lower4, upper4);
            __m512 const res003 = _mm512_add_ps(lower5, upper5);
            __m512 const res102 = _mm512_add_ps(lower14, upper14);
            __m512 const res103 = _mm512_add_ps(lower15, upper15);


            __m512 const upper6 =  _mm512_unpacklo_ps(res002, res003);
            __m512 const lower6 = _mm512_unpackhi_ps(res002, res003);
            __m512 const upper16 = _mm512_unpacklo_ps(res102, res103);
            __m512 const lower16 = _mm512_unpackhi_ps(res102, res103);
            __m512 const res011 = _mm512_add_ps(lower6, upper6);
            __m512 const res111 = _mm512_add_ps(lower16, upper16);

            mat2_ptr -= 4*compressed_sz;
            mat2_ptr2 = mat2_ptr + compressed_sz;
            mat2_ptr3 = mat2_ptr2 + compressed_sz;
            mat2_ptr4 = mat2_ptr3 + compressed_sz;
            mat2i_ptr -= (4/l)*compressed_sz;

            for (int k = 0; k < sz / compression_ratio; k += simd_ele_width)
            {
                __m512i idx_vec = _mm512_load_ps(mat2i_ptr + k);
                // TODO: this is for a fixed 2:4 4-wide vectorwide N:M sparsity.
                //       Should generate masks at runtime for parameterization.
                const __m512i move_mask = _mm512_set_epi32(0x1C, 0x1C, 0x18, 0x18, 0x14, 0x14, 0x10, 0x10, 0x0C, 0x0C, 0x08, 0x08, 0x04, 0x04, 0x00, 0x00);
                idx_vec = _mm512_add_epi32(idx_vec, move_mask);

                b_vec = _mm512_load_ps(mat2_ptr + k);
                b_vec2= _mm512_load_ps(mat2_ptr2+ k);
                b_vec3= _mm512_load_ps(mat2_ptr3+ k);
                b_vec4= _mm512_load_ps(mat2_ptr4+ k);

                a_vec_t11 = _mm512_load_ps(mat1_ptr + k*compression_ratio);
                a_vec_t12 = _mm512_load_ps(mat1_ptr + k*compression_ratio + simd_ele_width);
                a_vec_t21 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio);
                a_vec_t22 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio + simd_ele_width);
                a_vec = _mm512_permutex2var_ps(a_vec_t11, idx_vec, a_vec_t12);
                a_vec2= _mm512_permutex2var_ps(a_vec_t21, idx_vec, a_vec_t22);

                sums4 = _mm512_fmadd_ps(a_vec, b_vec, sums4);
                sums24 = _mm512_fmadd_ps(a_vec2, b_vec, sums24);

                sums5 = _mm512_fmadd_ps(a_vec, b_vec2, sums5);
                sums25 = _mm512_fmadd_ps(a_vec2, b_vec2, sums25);

                sums6 = _mm512_fmadd_ps(a_vec, b_vec3, sums6);
                sums26 = _mm512_fmadd_ps(a_vec2, b_vec3, sums26);

                sums7 = _mm512_fmadd_ps(a_vec, b_vec4, sums7);
                sums27 = _mm512_fmadd_ps(a_vec2, b_vec4, sums27);
            }

            __m512 const upper8 = _mm512_unpacklo_ps(sums4, sums6); // 0100 1110
            __m512 const lower8 = _mm512_unpackhi_ps(sums4, sums6);
            __m512 const upper9 = _mm512_unpacklo_ps(sums5, sums7); // 0100 1110
            __m512 const lower9 = _mm512_unpackhi_ps(sums5, sums7);
            __m512 const upper18 = _mm512_unpacklo_ps(sums24, sums26); // 0100 1110
            __m512 const lower18 = _mm512_unpackhi_ps(sums24, sums26);
            __m512 const upper19 = _mm512_unpacklo_ps(sums25, sums27); // 0100 1110
            __m512 const lower19 = _mm512_unpackhi_ps(sums25, sums27);
            __m512 const res004 = _mm512_add_ps(lower8, upper8);
            __m512 const res005 = _mm512_add_ps(lower9, upper9);
            __m512 const res104 = _mm512_add_ps(lower18, upper18);
            __m512 const res105 = _mm512_add_ps(lower19, upper19);

            __m512 const upperA =  _mm512_unpacklo_ps(res004, res005);
            __m512 const lowerA = _mm512_unpackhi_ps(res004, res005);
            __m512 const upper1A = _mm512_unpacklo_ps(res104, res105);
            __m512 const lower1A = _mm512_unpackhi_ps(res104, res105);
            __m512 const res012 = _mm512_add_ps(lowerA, upperA);
            __m512 const res112 = _mm512_add_ps(lower1A, upper1A);

            mat2_ptr += 8*compressed_sz;
            mat2_ptr2 = mat2_ptr + compressed_sz;
            mat2_ptr3 = mat2_ptr2 + compressed_sz;
            mat2_ptr4 = mat2_ptr3 + compressed_sz;
            mat2i_ptr += (8/l)*compressed_sz;

            for (int k = 0; k < sz / compression_ratio; k += simd_ele_width)
            {
                __m512i idx_vec = _mm512_load_ps(mat2i_ptr + k);
                // TODO: this is for a fixed 2:4 4-wide vectorwide N:M sparsity.
                //       Should generate masks at runtime for parameterization.
                const __m512i move_mask = _mm512_set_epi32(0x1C, 0x1C, 0x18, 0x18, 0x14, 0x14, 0x10, 0x10, 0x0C, 0x0C, 0x08, 0x08, 0x04, 0x04, 0x00, 0x00);
                idx_vec = _mm512_add_epi32(idx_vec, move_mask);

                b_vec = _mm512_load_ps(mat2_ptr + k);
                b_vec2= _mm512_load_ps(mat2_ptr2+ k);
                b_vec3= _mm512_load_ps(mat2_ptr3+ k);
                b_vec4= _mm512_load_ps(mat2_ptr4+ k);

                a_vec_t11 = _mm512_load_ps(mat1_ptr + k*compression_ratio);
                a_vec_t12 = _mm512_load_ps(mat1_ptr + k*compression_ratio + simd_ele_width);
                a_vec_t21 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio);
                a_vec_t22 = _mm512_load_ps(mat1_ptr2+ k*compression_ratio + simd_ele_width);
                a_vec = _mm512_permutex2var_ps(a_vec_t11, idx_vec, a_vec_t12);
                a_vec2= _mm512_permutex2var_ps(a_vec_t21, idx_vec, a_vec_t22);

                sumsC = _mm512_fmadd_ps(a_vec, b_vec, sumsC);
                sums2C = _mm512_fmadd_ps(a_vec2, b_vec, sums2C);

                sumsD = _mm512_fmadd_ps(a_vec, b_vec2, sumsD);
                sums2D = _mm512_fmadd_ps(a_vec2, b_vec2, sums2D);

                sumsE = _mm512_fmadd_ps(a_vec, b_vec3, sumsE);
                sums2E = _mm512_fmadd_ps(a_vec2, b_vec3, sums2E);

                sumsF = _mm512_fmadd_ps(a_vec, b_vec4, sumsF);
                sums2F = _mm512_fmadd_ps(a_vec2, b_vec4, sums2F);
            }

            __m512 const upperC = _mm512_unpacklo_ps(sumsC, sumsE); 
            __m512 const lowerC = _mm512_unpackhi_ps(sumsC, sumsE);
            __m512 const upperD = _mm512_unpacklo_ps(sumsD, sumsF); 
            __m512 const lowerD = _mm512_unpackhi_ps(sumsD, sumsF);
            __m512 const upper1C = _mm512_unpacklo_ps(sums2C, sums2E); 
            __m512 const lower1C = _mm512_unpackhi_ps(sums2C, sums2E);
            __m512 const upper1D = _mm512_unpacklo_ps(sums2D, sums2F); 
            __m512 const lower1D = _mm512_unpackhi_ps(sums2D, sums2F);
            __m512 const res006 = _mm512_add_ps(lowerC, upperC);
            __m512 const res007 = _mm512_add_ps(lowerD, upperD);
            __m512 const res106 = _mm512_add_ps(lower1C, upper1C);
            __m512 const res107 = _mm512_add_ps(lower1D, upper1D);

            __m512 const upperE =  _mm512_unpacklo_ps(res006, res007); 
            __m512 const lowerE = _mm512_unpackhi_ps(res006, res007); 
            __m512 const upper1E = _mm512_unpacklo_ps(res106, res107); 
            __m512 const lower1E = _mm512_unpackhi_ps(res106, res107); 
            __m512 const res013 = _mm512_add_ps(lowerE, upperE);
            __m512 const res113 = _mm512_add_ps(lower1E, upper1E);

            __m512 const upper11118 = _mm512_load_ps(dst_ptr2);
            __m512 const upper118 = _mm512_load_ps(dst_ptr);

            __m512i const permutemask0 = _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08);
            __m512i const permutemask2 = _mm512_set_epi32(0x1B, 0x1A, 0x19, 0x18, 0x0F, 0x0E, 0x0D, 0x0C, 0x13, 0x12, 0x11, 0x10, 0x07, 0x06, 0x05, 0x04);
            __m512 const upperF = _mm512_permutex2var_ps(res012, permutemask0, res013);
            __m512 const upper1F = _mm512_permutex2var_ps(res112, permutemask0, res113);
            __m512 const upper7 = _mm512_permutex2var_ps(res010, permutemask0, res011);
            __m512 const upper17 = _mm512_permutex2var_ps(res110, permutemask0, res111);
            __m512 const lower7 = _mm512_mask_blend_ps(0xFF00, res010, res011);
            __m512 const lower17 = _mm512_mask_blend_ps(0xFF00, res110, res111);
            __m512 const lowerF = _mm512_mask_blend_ps(0xFF00, res012, res013);
            __m512 const lower1F = _mm512_mask_blend_ps(0xFF00, res112, res113);
            __m512 const res021 = _mm512_add_ps(lowerF, upperF);
            __m512 const res121 = _mm512_add_ps(lower1F, upper1F);
            __m512 const res020 = _mm512_add_ps(lower7, upper7);
            __m512 const res120 = _mm512_add_ps(lower17, upper17);

            __m512 const lower0 = _mm512_permutex2var_ps(res020, permutemask2, res021);
            __m512 const lower10 = _mm512_permutex2var_ps(res120, permutemask2, res121);
            __m512 const upper0 = _mm512_mask_blend_ps(0xF0F0, res020, res021);
            __m512 const upper10 = _mm512_mask_blend_ps(0xF0F0, res120, res121);
            dst2 = _mm512_add_ps(lower0, upper0);
            dst3 = _mm512_add_ps(lower10, upper10);

        dst2 = _mm512_add_ps(dst2, upper118);
        dst3 = _mm512_add_ps(dst3, upper11118);
        _mm512_store_ps(dst_ptr, dst2);
        _mm512_store_ps(dst_ptr2, dst3);
        dst_ptr += simd_ele_width;
        dst_ptr2 += simd_ele_width;
            //float sum0 = _mm512_reduce_add_ps(sums0);

            //*(dst + i*sz+ j) = sum0;
        }
    }
    
    return NULL;
}
#endif // USE_AVX512

#endif  // __GEMM_H__
