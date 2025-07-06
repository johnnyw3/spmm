import random, sys

def gen_matrix(fname, n):
    '''
    Generates a random n x n matrix with numbers between 0 and 100, outputting
    to the given fname
    '''
    res = []

    for idx_x in range(n):
        for idx_y in range(n):
            res.append(random.randint(0, 100))


    with open(fname, "w") as fp:
        fp.write("%d\n" % (n))
        for num in res:
            fp.write("%d " % (num))

def gen_vnm_sparse_matrix(fname, n, m, l, sz):
    '''
    Generates a random vector-based n:m sparsity matrix of size sz x sz with
    vector length, outputting to the given fname
    '''
    assert( not sz % l and "Size must be a multiple of vector length" )
    res = [ [] for row in range(sz) ]

    for outer_x in range(0, sz, l):
        for outer_y in range(0, sz, m):
            nonzero_vecs = set(random.sample(range(m), n))
            for inner_y in range(m):
                 if inner_y not in nonzero_vecs:
                    res[outer_y + inner_y].extend( [0] * l )
                 else:
                    res[outer_y + inner_y].extend( [ random.randint(1, 9) for idx in range(l) ] )

    with open(fname, "w") as fp:
        fp.write("%d\n" % (sz))
        for row in res:
            for num in row:
                fp.write("%d " % (num))
            fp.write("\n")

if __name__ == '__main__':
    #gen_matrix(sys.argv[1], int(sys.argv[2]))
    gen_vnm_sparse_matrix(sys.argv[1], *[int(arg) for arg in sys.argv[2:]])
