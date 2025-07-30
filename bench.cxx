#include <iostream>
#include <chrono>
#include <stdint.h>
#include <cblas.h>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include "simd_common.h"
#include "spmm.h"
#include "bench.h"

int main(int argv, char **argc)
{
    printf("Using %d-wide SIMD, %d threads\n", SIMD_WIDTH*8, NUM_THREADS);

    float *mat1, *mat2;
    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);
    int n_blk = atoi(argc[3]), m = atoi(argc[4]), l = atoi(argc[5]);  
    std::size_t n_large = n;

    // OpenBLAS (GEMM) benchmark
    float *dst_cblas = (float*)aligned_alloc(64, (sizeof(float) * n * n));
    std::size_t time_sum_blas = 0;
    int num_runs_cblas = 10;
    for (int idx = 0; idx < num_runs_cblas; ++idx)
    {
        memset(dst_cblas, 0, sizeof(float) * n * n);

        auto const start = std::chrono::high_resolution_clock::now();
        cblas_semm(mat1, mat2, dst_cblas, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum_blas += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }
    //print_mat(dst_cblas, n, n);
    double gflops = get_gflops(time_sum_blas, num_runs_cblas*2*n_large*n_large*n_large);
    printf("OpenBLAS time: %lfs, gflops: %f\n", time_sum_blas*1.0/num_runs_cblas/US_PER_S, gflops);
    //print_mat(dst_cblas, n);

    // Eigen (SpMM) benchmark
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve( n * n * n_blk / m);
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
        {
            if ( *(mat2 + idx*n + jdx) )
                triplets.push_back( Eigen::Triplet<float>(idx, jdx, *(mat2 + idx*n + jdx) ) );
        }
    }
    Eigen::SparseMatrix<float> mat2sp(n, n);
    mat2sp.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat1dn(mat1, n, n);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dstdn(n, n);

    std::size_t time_sum_eigen = 0;
    int num_runs_eigen = 3;
    for (int idx = 0; idx < num_runs_eigen; ++idx)
    {
        auto const start = std::chrono::high_resolution_clock::now();
        dstdn = mat1dn * mat2sp;

        auto const end = std::chrono::high_resolution_clock::now();
        verify_matrix(dst_cblas, dstdn.data(), n);
        //printf("done\n");
        time_sum_eigen += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }
    //print_mat(dst_cblas, n, n);
    gflops = get_gflops(time_sum_eigen, num_runs_eigen*2*n_large*n_large*n_large);
    printf("Eigen (sparse) time: %lfs, gflops: %f\n", time_sum_eigen*1.0/num_runs_eigen/US_PER_S, gflops);

    // Now actually run our code.

    float *dst = (float*)aligned_alloc(64, sizeof(float) * n * n);
    float *mat2c = (float*)aligned_alloc(64, sizeof(float) * n * n * n_blk / m);
    int *dst_idx = (int*)aligned_alloc(64, sizeof(int) * n * n / l * n_blk / m);
    memset(dst, 0, sizeof(float) * n * n);

    std::size_t time_sum = 0;
    int num_runs = 10;

    for (int idx = 0; idx < num_runs; ++idx)
    {
        // PREP WORK: other papers didn't include this work in their time calculations...
        squash_matrix(mat2, mat2c, dst_idx, n_blk, m, l, n); 
        cpu_transpose(mat2c, n, n * n_blk / m); 
        cpu_transpose(dst_idx, n / l , n * n_blk / m); 

        auto const start = std::chrono::high_resolution_clock::now();
        simd_spmm(mat1, mat2c, dst_idx, dst, n, n_blk, m, l);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //printf("done\n");

        //print_mat(dst, n, n);
        //printf("Original matrix: \n");
        //print_mat(mat2, n, n );
        //printf("Compressed matrix: \n");
        //print_mat(mat2c, n * n_blk / m, n);
        //print_mat(mat2c, n, n * n_blk / m);
        //printf("Index matrix: \n");
        // Already transposed.
        //print_mat(dst_idx, n * n_blk / m, n / l);
        //print_mat(dst_idx, n/l, n * n_blk / m);
        verify_matrix(dst_cblas, dst, n);

        // zero dst so that the algorithm actually needs to successfully run for
        // multiple tests to pass.
        memset(dst, 0, sizeof(float) * n * n);
    }

    printf("\n");

    gflops = get_gflops(time_sum, num_runs*2*n_large*n_large*n_large);
    printf("Avg time: %lfs, gflops: %f\n", time_sum*1.0/num_runs/US_PER_S, gflops);
    
    free(mat1);
    free(mat2);
    free(mat2c);
    free(dst);
    free(dst_idx);
    return 0;

}

int read_mat(char *fname, int *n, float **dst)
{
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        perror("Error opening matrix data file");
        return 1;
    }

    fscanf(fp, "%d", n);
    std::cout << *n << "\n";

    *dst = (float*)aligned_alloc(64, (sizeof(float) * *n * *n));

    for (int idx = 0; idx < *n**n; ++idx)
    {
        fscanf(fp, "%f", *dst + idx);
    }

    fclose(fp);
    return 0;
}

void print_mat(int *mat, int n_col, int n_row)
{
    for (int idx = 0; idx < n_row; ++idx)
    {
        for (int jdx = 0; jdx < n_col; ++jdx)
            printf("%d ", *(mat + idx*n_col + jdx));
        printf("\n");
    }

}


void print_mat(int *mat, int n)
{
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
            printf("%d ", *(mat + idx*n + jdx));
        printf("\n");
    }

}

void print_mat(float *mat, int n_col, int n_row)
{
    for (int idx = 0; idx < n_row; ++idx)
    {
        for (int jdx = 0; jdx < n_col; ++jdx)
            printf("%.0f ", *(mat + idx*n_col + jdx));
        printf("\n");
    }

}

void print_mat(float *mat, int n)
{
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
            printf("%.0f ", *(mat + idx*n + jdx));
        printf("\n");
    }

}

double get_gflops(std::size_t us, std::size_t n)
{
    double s = us*1.0 / US_PER_S;
    return n / s / GIGA;
}

void cblas_semm(float *mat1, float *mat2, float *dst, int n)
{
    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                 mat1, n, mat2, n, 1.0, dst, n );
}

void verify_matrix(float *exp, float *act, int n)
{
    int incorrect = 0;
    for (int idx_x = 0; idx_x < n; ++idx_x)
    {
        for (int idx_y = 0; idx_y < n; ++idx_y)
        {
            float exp_val = *(exp + idx_y*n + idx_x);
            float act_val = *(act + idx_y*n + idx_x);

            if (exp_val != act_val)
            {
                printf("difference at: (%d, %d). exp: %.2f, act: %.2f\n", idx_x, idx_y, exp_val, act_val);
                incorrect = 1;
            }
        }
    }

    if (!incorrect)
        printf("Matricies are the same\n");
}
