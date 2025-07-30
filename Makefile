# for logging etc
TEMP_FNAME=$(shell date +%FT%H:%M:%S)_log.txt

NUM_THREADS?=$(shell nproc)

TARGET?=tigerlake
CXXFLAGS?=-march=$(TARGET) -O3 -flto -g -fsave-optimization-record -DNUM_THREADS=$(NUM_THREADS) -DTARGET=$(TARGET)
CC=clang++

all: bench

build/libspmm.a: spmm.cxx spmm.h simd_common.h
	@mkdir -p build
	$(CC) -o build/spmm.o -c spmm.cxx $(CXXFLAGS)
	ar rcs build/libspmm.a build/spmm.o

bench: build/libspmm.a bench.cxx bench.h
	$(CC) -o build/bench bench.cxx -Lbuild -lspmm -lopenblas -I/usr/include/eigen3/ $(CXXFLAGS)

clean:
	rm -rf build
