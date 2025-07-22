# SpMM

High-performance CPU SpMM (N:M sparsity) algorithm accelerated using AVX.

TODOs:

 * SpMM-specific performance optimizations
 * Parameterized N and M (currently only works for 2:4 vectorwise sparsity)

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
