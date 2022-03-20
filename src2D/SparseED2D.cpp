#include "SparseED2D.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "Time.h"
#include "mkl.h"
typedef long long ll;

SparseED2D::SparseED2D(int Nx_, int Ny_, double t_, double U_, double T_, double mu_, int Ncut_)
    : Nx(Nx_),
      Ny(Ny_),
      Ns(Nx_ * Ny_),
      NN((Nx_ * Ny_ + 1) * Nx_ * Ny_),
      NNN((Nx_ * Ny_ + 1) * (Nx_ * Ny_ + 1) * Nx_ * Ny_),
      DIM(1 << (Nx_ * Ny_)),
      t(t_),
      U(U_),
      T(T_),
      mu(mu_),
      Ncut(Ncut_) {
    double TwoPiOverNx = 6.28318530717959 / Nx;
    double TwoPiOverNy = 6.28318530717959 / Ny;
    double *epsilon = new double[Ns];
    for (int k = 0; k < Ns; k++) {
        int kx = k / Ny, ky = k % Ny;
        epsilon[k] = -2. * t * (std::cos(TwoPiOverNx * kx) + std::cos(TwoPiOverNy * ky)) - mu;
    }

    Nparticle = new int[DIM];
    Momentum = new int[DIM];
    Ek = new double[DIM];
    Nparticle[0] = 0;
    Momentum[0] = 0;
    Ek[0] = 0;
    for (int k = 0; k < Ns; k++) {
        uint kstate = 1 << k;
        int kx = k / Ny, ky = k % Ny;
        for (uint state = 0; state < kstate; state++) {
            Nparticle[state ^ kstate] = Nparticle[state] + 1;
            Momentum[state ^ kstate] = (Momentum[state] / Ny + kx) % Nx * Ny + (Momentum[state] % Ny + ky) % Ny;
            Ek[state ^ kstate] = Ek[state] + epsilon[k];
        }
    }
    delete[] epsilon;

    SubDim = new int[NN];
    int *block = new int[DIM];
    int *index = new int[DIM];
    memset(SubDim, 0, (Ns + 1) * Ns * sizeof(int));
    for (uint state = 0; state < DIM; state++) {
        int blkid = Nparticle[state] * Ns + Momentum[state];
        block[state] = blkid;
        index[state] = SubDim[blkid]++;
    }

    SubBase = new uint *[NN];
    memset(SubBase, 0, NN * sizeof(uint *));
    for (int blkid = 0; blkid < NN; blkid++) {
        SubBase[blkid] = new uint[SubDim[blkid]];
    }

    for (uint state = 0; state < DIM; state++) {
        SubBase[block[state]][index[state]] = state;
    }

    delete[] block;
    delete[] index;

    dim = new int[NNN];
    Ne = new int[NNN];
    E = new double *[NNN];
    V = new double *[NNN];
    memset(dim, 0, NNN * sizeof(int));
    memset(E, 0, NNN * sizeof(double *));
    memset(V, 0, NNN * sizeof(double *));
    double UOverNs = U / Ns;
    for (int blkid = 0; blkid < NNN; blkid++) {
        int Nu = blkid / NN, Nd = blkid % NN / Ns, p = blkid % Ns;
        for (int ku = 0; ku < Ns; ku++) {
            int kd = (Nx + p / Ny - ku / Ny) % Nx * Ny + (Ny + p % Ny - ku % Ny) % Ny;
            dim[blkid] += SubDim[Nu * Ns + ku] * SubDim[Nd * Ns + kd];
        }
        if (dim[blkid] == 0) continue;
        std::cout << Time << "---Begin  block " << blkid << "---" << std::endl;
        std::cout << "Nu = " << Nu << ", Nd = " << Nd << ", px = " << p / Ny << ", py = " << p % Ny << std::endl;
        std::cout << "dim = " << dim[blkid] << std::endl;

        uint *base = new uint[dim[blkid]];
        int idx = 0;
        for (int ku = 0; ku < Ns; ku++) {
            int kd = (Nx + p / Ny - ku / Ny) % Nx * Ny + (Ny + p % Ny - ku % Ny) % Ny;
            int blku = Nu * Ns + ku, blkd = Nd * Ns + kd;
            for (int idxd = 0; idxd < SubDim[blkd]; idxd++) {
                for (int idxu = 0; idxu < SubDim[blku]; idxu++) {
                    base[idx++] = (SubBase[blkd][idxd] << Ns) ^ SubBase[blku][idxu];
                }
            }
        }
        std::sort(base, base + dim[blkid]);

        if (dim[blkid] < Ncut / 2 * 3) {
            // Use dense Matrix eigen solver: LAPACKE_dsyev
            double *H = new double[dim[blkid] * dim[blkid]];
            memset(H, 0, dim[blkid] * dim[blkid] * sizeof(double));

            for (int ridx = 0; ridx < dim[blkid]; ridx++) {
                uint rupstate = base[ridx] % DIM;
                uint rdnstate = base[ridx] >> Ns;

                // 1. diagonal 2rd order
                H[ridx * dim[blkid] + ridx] = Ek[rupstate] + Ek[rdnstate];

                // 2. diagonal 4th order k1 == k2 && k3 == k4
                for (int k1 = 0; k1 < Ns; k1++) {
                    if (((rupstate >> k1) & 1) == 0) continue;
                    for (int k3 = 0; k3 < Ns; k3++) {
                        if (((rdnstate >> k3) & 1) == 0) continue;
                        H[ridx * dim[blkid] + ridx] += UOverNs;
                    }
                }

                // 3. off diagonal
                for (int k1 = 0; k1 < Ns; k1++) {
                    for (int k2 = 0; k2 < Ns; k2++) {
                        if (((rupstate >> k1) & 1) || !((rupstate >> k2) & 1)) continue;
                        uint lupstate = (rupstate ^ (1 << k1)) ^ (1 << k2);
                        int ones12 = k1 < k2 ? Nparticle[rupstate >> (k1 + 1)] - Nparticle[rupstate >> k2]
                                             : Nparticle[rupstate >> (k2 + 1)] - Nparticle[rupstate >> k1];
                        bool minusup = ones12 & 1;

                        for (int k3 = 0; k3 < Ns; k3++) {
                            int k4 = (Nx + k1 / Ny + k3 / Ny - k2 / Ny) % Nx * Ny + (Ny + k1 % Ny + k3 % Ny - k2 % Ny) % Ny;
                            if (((rdnstate >> k3) & 1) || !((rdnstate >> k4) & 1)) continue;

                            int ones34 = k3 < k4 ? Nparticle[rdnstate >> (k3 + 1)] - Nparticle[rdnstate >> k4]
                                                 : Nparticle[rdnstate >> (k4 + 1)] - Nparticle[rdnstate >> k3];
                            bool minusdown = ones34 & 1;
                            uint ldnstate = (rdnstate ^ (1 << k3)) ^ (1 << k4);
                            uint lstate = (ldnstate << Ns) ^ lupstate;
                            int lidx = std::lower_bound(base, base + dim[blkid], lstate) - base;
                            if (minusup == minusdown)
                                H[lidx * dim[blkid] + ridx] += UOverNs;
                            else
                                H[lidx * dim[blkid] + ridx] -= UOverNs;
                        }
                    }
                }
            }

            E[blkid] = new double[dim[blkid]];
            Ne[blkid] = dim[blkid];
            LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', dim[blkid], H, dim[blkid], E[blkid]);
            delete[] H;
        }
        else {
            // Use sparce Matrix eigen solver: mkl_sparse_d_ev
            ll nnzestimate = ll(dim[blkid]) * Nu * (Ns - Nu + 1) * std::min(Nd, Ns - Nd + 1);
            if (nnzestimate == 0) nnzestimate = dim[blkid];
            MKL_INT *ia = new MKL_INT[dim[blkid] + 1];
            MKL_INT *ja = new MKL_INT[nnzestimate];
            double *a = new double[nnzestimate];

            ll nnz = 0;
            for (int ridx = 0; ridx < dim[blkid]; ridx++) {
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
                            int k4 = (Nx + k1 / Ny + k3 / Ny - k2 / Ny) % Nx * Ny + (Ny + k1 % Ny + k3 % Ny - k2 % Ny) % Ny;
                            if (((rdnstate >> k3) & 1) || !((rdnstate >> k4) & 1)) continue;

                            int ones34 = k3 < k4 ? Nparticle[rdnstate >> (k3 + 1)] - Nparticle[rdnstate >> k4]
                                                 : Nparticle[rdnstate >> (k4 + 1)] - Nparticle[rdnstate >> k3];
                            bool minusdown = ones34 & 1;
                            uint ldnstate = (rdnstate ^ (1 << k3) ^ (1 << k4));
                            uint lstate = (ldnstate << Ns) ^ lupstate;
                            int lidx = std::lower_bound(base, base + dim[blkid], lstate) - base;
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
            ia[dim[blkid]] = nnz;
            std::cout << "nnz = " << nnz << std::endl;

            if (nnz == dim[blkid]) {
                E[blkid] = new double[Ncut];
                std::sort(a, a + dim[blkid]);
                memcpy(E[blkid], a, Ncut * sizeof(double));
                Ne[blkid] = Ncut;
            }
            else {
                sparse_matrix_t A;
                struct matrix_descr descr = {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
                sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, dim[blkid], dim[blkid], ia, ia + 1, ja, a);
                if (status != 0) {
                    std::cout << "Routine mkl_sparse_d_create_csr returns code of ERROR: " << status << std::endl;
                }
                mkl_sparse_order(A);

                MKL_INT pm[128];
                MKL_INT k0 = Ncut;
                MKL_INT k;

                E[blkid] = new double[k0];
                double *res = new double[k0];
                mkl_sparse_ee_init(pm);
                pm[1] = 8;
                pm[2] = 1;
                pm[6] = 0;
                pm[7] = 1;

                sparse_status_t info = mkl_sparse_d_ev("S", pm, A, descr, k0, &k, E[blkid], V[blkid], res);
                if (info != 0) {
                    std::cout << "Routine mkl_sparse_d_ev returns code of ERROR: " << info << std::endl;
                    exit(1);
                }

                Ne[blkid] = k;

                delete[] res;

                mkl_sparse_destroy(A);
            }

            delete[] a;
            delete[] ja;
            delete[] ia;
        }

        delete[] base;

        std::cout << Time << "---Finish block " << blkid << "---" << std::endl;
    }
}

SparseED2D::~SparseED2D() {
    for (int blkid = 0; blkid < NNN; blkid++) {
        delete[] V[blkid];
        delete[] E[blkid];
    }

    delete[] V;
    delete[] E;

    delete[] Ne;
    delete[] dim;

    for (int blkid = 0; blkid < NN; blkid++) {
        delete[] SubBase[blkid];
    }

    delete[] SubBase;
    delete[] SubDim;
    delete[] Ek;
    delete[] Momentum;
    delete[] Nparticle;
}

double SparseED2D::density() const {
    // find ground energy
    double groundE = 0;
    for (int blkid = 0; blkid < NNN; blkid++) {
        if (dim[blkid] == 0) continue;
        groundE = std::min(groundE, E[blkid][0]);
    }

    double den = 0, Z = 0.;
    for (int blkid = 0; blkid < NNN; blkid++) {
        int Nu = blkid / NN, Nd = blkid % NN / Ns;
        for (int idx = 0; idx < Ne[blkid]; idx++) {
            den += (Nu + Nd) * std::exp(-(E[blkid][idx] - groundE) / T);
            Z += std::exp(-(E[blkid][idx] - groundE) / T);
        }
    }

    return den / (double(Ns) * Z);
}

void SparseED2D::setT(double T_) { T = T_; }
void SparseED2D::setmu(double mu_) {
    for (int blkid = 0; blkid < NNN; blkid++) {
        double che_diff = (blkid / NN + blkid % NN / Ns) * (mu_ - mu);
        for (int idx = 0; idx < Ne[blkid]; idx++) {
            E[blkid][idx] -= che_diff;
        }
    }
    mu = mu_;
}