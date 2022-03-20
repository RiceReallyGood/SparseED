#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "mkl.h"
using ll = long long;
int main() {
    int Ns = 10;
    int DIM = 1 << Ns;
    double t = 1., U = 4., mu = U / 2;

    double TwoPiOverNs = 6.28318530717959 / Ns;
    double *epsilon = new double[Ns];
    for (int k = 0; k < Ns; k++) {
        epsilon[k] = -2. * t * std::cos(TwoPiOverNs * k) - mu;
    }

    int *Nparticle = new int[DIM];
    int *Momentum = new int[DIM];
    double *Ek = new double[DIM];
    Momentum[0] = 0;
    Nparticle[0] = 0;
    Ek[0] = 0.;
    for (int k = 0; k < Ns; k++) {
        uint kstate = 1 << k;
        for (uint state = 0; state < kstate; state++) {
            Nparticle[state ^ kstate] = Nparticle[state] + 1;
            Momentum[state ^ kstate] = (Momentum[state] + k) % Ns;
            Ek[state ^ kstate] = Ek[state] + epsilon[k];
        }
    }

    delete[] epsilon;

    int *SubDim = new int[(Ns + 1) * Ns];
    int *block = new int[DIM];
    int *index = new int[DIM];
    memset(SubDim, 0, (Ns + 1) * Ns * sizeof(int));
    for (uint state = 0; state < DIM; state++) {
        int blkid = Nparticle[state] * Ns + Momentum[state];
        block[state] = blkid;
        index[state] = SubDim[blkid]++;
    }

    uint **SubBase = new uint *[(Ns + 1) * Ns];
    for (int blkid = 0; blkid < Ns * (Ns + 1); blkid++) {
        SubBase[blkid] = new uint[SubDim[blkid]];
    }

    for (uint state = 0; state < DIM; state++) {
        SubBase[block[state]][index[state]] = state;
    }

    delete[] block;
    delete[] index;

    int Nu = Ns / 2, Nd = Ns / 2;
    int p = 0;
    int dim = 0;
    for (int ku = 0; ku < Ns; ku++) {
        int kd = (Ns + p - ku) % Ns;
        dim += SubDim[Nu * Ns + ku] * SubDim[Nd * Ns + kd];
    }

    std::cout << "dim = " << dim << std::endl;

    uint *base = new uint[dim];
    int idx = 0;
    for (int ku = 0; ku < Ns; ku++) {
        int kd = (Ns + p - ku) % Ns;
        int blku = Nu * Ns + ku, blkd = Nd * Ns + kd;
        for (int idxd = 0; idxd < SubDim[blkd]; idxd++) {
            for (int idxu = 0; idxu < SubDim[blku]; idxu++) {
                base[idx++] = (SubBase[blkd][idxd] << Ns) ^ SubBase[blku][idxu];
            }
        }
    }

    std::sort(base, base + dim);

    std::cout << "find out base" << std::endl;

    ll nnzestimate = ll(dim) * Nu * (Ns - Nu + 1) * std::min(Nd, Ns - Nd + 1);
    MKL_INT *ia = new MKL_INT[dim + 1];
    MKL_INT *ja = new MKL_INT[nnzestimate];
    double *a = new double[nnzestimate];

    ll nnz = 0;
    double UOverNs = U / Ns;
    for (int ridx = 0; ridx < dim; ridx++) {
        ia[ridx] = nnz;
        uint rupstate = base[ridx] % DIM;
        uint rdnstate = base[ridx] >> Ns;
        // 1. diagonal 2rd order
        a[nnz] = Ek[rupstate] + Ek[rdnstate];
        ja[nnz] = ridx;

        // 2. diagonal 4th order k1 == k2 && k3 == k4
        for (int k1 = 0; k1 < Ns; k1++) {
            if (((rupstate >> k1) & 1) == 0) continue;
            for (int k3 = 0; k3 < Ns; k3++) {
                if (((rdnstate >> k3) & 1) == 0) continue;
                a[nnz] += UOverNs;
            }
        }

        nnz++;

        // 3. off diagonal
        for (int k1 = 0; k1 < Ns; k1++) {
            for (int k2 = 0; k2 < Ns; k2++) {
                if (((rupstate >> k1) & 1) || !((rupstate >> k2) & 1)) continue;
                uint lupstate = (rupstate ^ (1 << k1)) ^ (1 << k2);
                int ones12 = k1 < k2 ? Nparticle[rupstate >> (k1 + 1)] - Nparticle[rupstate >> k2]
                                     : Nparticle[rupstate >> (k2 + 1)] - Nparticle[rupstate >> k1];
                bool minusup = ones12 & 1;

                for (int k3 = 0; k3 < Ns; k3++) {
                    int k4 = (Ns + k1 + k3 - k2) % Ns;
                    if (((rdnstate >> k3) & 1) || !((rdnstate >> k4) & 1)) continue;

                    int ones34 = k3 < k4 ? Nparticle[rdnstate >> (k3 + 1)] - Nparticle[rdnstate >> k4]
                                         : Nparticle[rdnstate >> (k4 + 1)] - Nparticle[rdnstate >> k3];
                    bool minusdown = ones34 & 1;
                    uint ldnstate = (rdnstate ^ (1 << k3) ^ (1 << k4));
                    uint lstate = (ldnstate << Ns) ^ lupstate;
                    int lidx = std::lower_bound(base, base + dim, lstate) - base;
                    if (minusup == minusdown) {
                        ja[nnz] = lidx;
                        a[nnz] = UOverNs;
                    }
                    else {
                        ja[nnz] = lidx;
                        a[nnz] = -UOverNs;
                    }
                    nnz++;
                }
            }
        }
    }
    ia[dim] = nnz;
    std::cout << "nnz = " << nnz << std::endl;

    // for (int i = 0; i < dim; i++) {
    //     for (int j = ia[i]; j < ia[i + 1]; j++) {
    //         std::cout << '(' << ja[j] << ", " << a[j] << ") ";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "End write matrix!" << std::endl;

    sparse_matrix_t A;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, dim, dim, ia, ia + 1, ja, a);
    if (status != 0) {
        std::cout << "Routine mkl_sparse_d_create_csr returns code of ERROR: " << status << std::endl;
    }
    mkl_sparse_order(A);

    std::cout << "End create sparse matrix!" << std::endl;

    MKL_INT pm[128];
    MKL_INT k0 = 100;
    MKL_INT k;
    double *E = new double[k0];
    double *res = new double[k0];
    double *V = nullptr;
    mkl_sparse_ee_init(pm);
    pm[1] = 8;
    pm[2] = 1;
    pm[6] = 0;

    std::cout << "Begin calculate!" << std::endl;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    int info = mkl_sparse_d_ev("S", pm, A, descr, k0, &k, E, V, res);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Elapsed time: " << diff.count() << "s" << std::endl;

    for (int i = 0; i < k; i++) {
        std::cout << E[i] << std::endl;
    }

    if (info != 0) {
        std::cout << "Routine mkl_sparse_d_ev returns code of ERROR: " << info << std::endl;
        return 1;
    }

    delete[] V;
    delete[] res;
    delete[] E;

    delete[] a;
    delete[] ja;
    delete[] ia;
}