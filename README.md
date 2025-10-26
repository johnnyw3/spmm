# SpMM

High-performance CPU SpMM (N:M sparsity) algorithm accelerated using AVX.

TODOs:

 * More performance optimization?
 * Parameterized N and M (currently only works for 2:4 sparsity)

# Benchmarks

Metrics are in GFLOPs; speedups are compared to either OpenBLAS or MKL,
whichever is faster. Types are `fp32`.

**Single-threaded, n=4096** average of 10 runs

| Kernel | CPU | This algorithm | BLAS | Speedup |
AVX-512 | **Tiger Lake** i5-1135G7 | 146 | 122 | 1.20 |
AVX-512 | **Granite Rapids** Xeon 6972P |  158 | --- | --- |

**Multithreaded, n=4096** average of 10 runs

| Kernel | CPU | Threads |  This algorithm | BLAS | Speedup |
|:-------|:----|--------:|---------------:|---------:|:------------------|
AVX-512 | **Tiger Lake** i5-1135G7 | 4 | 560 | 465 | 1.20 |
AVX-512 | **Granite Rapids** Xeon 6972P | 4 |  629 | --- | ---  |

# Prerequisites

## Hardware

A modern x86 processor supporting AVX2 or AVX-512 must be used.

## Software

For the library:

* `pthreads` support (for multithreading)
* A version of `make` that supports the `shell` directive

For the `bench` program:

* OpenBLAS (for verifying correctness and providing a baseline to compare against)

# Build

To build, use the given makefile, specifying your march. For example:

```bash
$ make TARGET=skylake
```

When compiling for a target that supports AVX-512, AVX-512 operations will 
automatically be used.

By default, multithreading is enabled using as many threads as logical
processors on your system. To change the number of threads, pass the
`NUM_THREADS=<num>` option:

```bash
$ make TARGET=skylake NUM_THREADS=4
```

You'll need to `make clean` first before changing configurations.
