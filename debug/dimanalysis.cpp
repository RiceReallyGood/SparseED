#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "mkl.h"
using namespace std;

int onesbetween(int state, int* ones, int k1, int k2) {
    return k1 < k2 ? ones[state >> (k1 + 1)] - ones[state >> k2] : ones[state >> (k2 + 1)] - ones[state >> k1];
}

void printmat(const std::string& name, int m, int n, int lda, const double* mat, int prec) {
    std::ios::fmtflags OldFlags = std::cout.flags();
    std::cout << name << " :" << std::endl;
    std::cout.unsetf(std::ios::floatfield);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(prec);

    int width = prec + 3;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(width) << mat[i * lda + j];
            if (j != n - 1) std::cout << "   ";
        }
        std::cout << std::endl;
    }

    std::cout.flags(OldFlags);
    std::cout.precision(6);
}

int main() {
    int Ns = 8;
    int NN = (Ns + 1) * Ns;
    int NNN = (Ns + 1) * (Ns + 1) * Ns;
    double t = 1., U = 4, mu = U / 2;
    int onespindim = 1 << Ns;
    int DIM = 1 << (2 * Ns);

    int* ones = new int[onespindim];
    ones[0] = 0;
    for (int n = 1; n < onespindim; n++) {
        ones[n] = ones[n & (n - 1)] + 1;
    }

    double TwoPiOverNs = 6.283185307179586 / Ns;
    double* epsilon = new double[Ns];
    for (int k = 0; k < Ns; k++) {
        epsilon[k] = -2. * t * std::cos(TwoPiOverNs * k) - mu;
    }

    int* dim = new int[NNN];
    memset(dim, 0, NNN * sizeof(int));
    int* index = new int[DIM];
    int* block = new int[DIM];
    double* Ek = new double[DIM];
    for (int state = 0; state < DIM; state++) {
        int upstate = state % onespindim, downstate = state / onespindim;
        int nu = ones[upstate], nd = ones[downstate];
        int p = 0;
        double ek = 0;
        for (int k = 0; k < Ns; k++) {
            int nk = (upstate & 1) + (downstate & 1);
            p += nk * k;
            ek += nk * epsilon[k];
            upstate >>= 1, downstate >>= 1;
            if (upstate == 0 && downstate == 0) break;
        }

        Ek[state] = ek;
        p %= Ns;
        int blkid = NN * nu + Ns * nd + p;
        block[state] = blkid;
        index[state] = dim[blkid]++;
    }

    delete[] epsilon;

    int** base = new int*[NNN];
    for (int blkid = 0; blkid < NNN; blkid++) {
        if (dim[blkid] > 0) {
            base[blkid] = new int[dim[blkid]];
        }
    }
    for (int state = 0; state < DIM; state++) {
        base[block[state]][index[state]] = state;
    }

    delete[] block;

    double** H = new double*[NNN];
    double** egvalues = new double*[NNN];
    for (int blkid = 0; blkid < NNN; blkid++) {
        if (dim[blkid] > 0) {
            H[blkid] = new double[dim[blkid] * dim[blkid]];
            egvalues[blkid] = new double[dim[blkid]];
        }
    }

    double UOverNs = U / Ns;
    for (int blkid = 0; blkid < NNN; blkid++) {
        if (dim[blkid] == 0) continue;
        memset(H[blkid], 0, dim[blkid] * dim[blkid] * sizeof(double));

#pragma omp parallel for
        for (int ridx = 0; ridx < dim[blkid]; ridx++) {
            int rstate = base[blkid][ridx];
            int rupstate = rstate % onespindim, rdownstate = rstate / onespindim;

            // 1. diagonal 2rd order
            H[blkid][ridx * dim[blkid] + ridx] = Ek[rstate];

            // 2. diagonal 4th order k1 == k2 && k3 == k4
            for (int k1 = 0; k1 < Ns; k1++) {
                if (((rupstate >> k1) & 1) == 0) continue;
                for (int k3 = 0; k3 < Ns; k3++) {
                    if (((rdownstate >> k3) & 1) == 0) continue;
                    H[blkid][ridx * dim[blkid] + ridx] += UOverNs;
                }
            }

            // 3. off diagonal
            for (int k1 = 0; k1 < Ns; k1++) {
                for (int k2 = 0; k2 < Ns; k2++) {
                    if (((rupstate >> k1) & 1) || !((rupstate >> k2) & 1)) continue;
                    int lupstate = (rupstate ^ (1 << k1)) ^ (1 << k2);
                    bool minusup = onesbetween(rupstate, ones, k1, k2) & 1;

                    for (int k3 = 0; k3 < Ns; k3++) {
                        int k4 = (Ns + k1 + k3 - k2) % Ns;
                        if (((rdownstate >> k3) & 1) || !((rdownstate >> k4) & 1)) continue;

                        bool minusdown = onesbetween(rdownstate, ones, k3, k4) & 1;
                        int ldownstate = (rdownstate ^ (1 << k3) ^ (1 << k4));
                        int lstate = ldownstate * onespindim + lupstate;
                        if (minusup == minusdown)
                            H[blkid][index[lstate] * dim[blkid] + ridx] += UOverNs;
                        else
                            H[blkid][index[lstate] * dim[blkid] + ridx] -= UOverNs;
                    }
                }
            }
        }
        // if (blkid == 21) {
        //     printmat("H of the block", dim[blkid], dim[blkid], dim[blkid], H[blkid], 6);
        // }

        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', dim[blkid], H[blkid], dim[blkid], egvalues[blkid]);
    }

    // for (int blkid = 0; blkid < NNN; blkid++) {
    //     std::cout << "blockid = " << blkid << ", Nup = " << blkid / NN << ", Ndn = " << blkid % NN / Ns
    //               << ", k = " << blkid % Ns << std::endl;
    //     std::cout << "Dimension of this block = " << dim[blkid] << std::endl;

    //     printmat("eigen values of the block", 1, dim[blkid], dim[blkid], egvalues[blkid], 6);
    //     std::cout << std::string(80, '=') << std::endl;
    // }
    int targetblkid = Ns / 2 * (Ns + 1) * Ns + Ns / 2 * Ns;
    for (int i = 0; i < 100; i++) {
        std::cout << egvalues[targetblkid][i] << std::endl;
    }

    for (int blkid = 0; blkid < NNN; blkid++) {
        if (dim[blkid] > 0) {
            delete[] base[blkid];
            delete[] H[blkid];
            delete[] egvalues[blkid];
        }
    }

    delete[] base;
    delete[] H;
    delete[] egvalues;

    delete[] ones;
    delete[] dim;
    delete[] index;
}