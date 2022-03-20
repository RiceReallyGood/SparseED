#include <iomanip>
#include <iostream>
#include <string>

#include "mkl.h"
#include "mkl_solvers_ee.h"

void printmat(const std::string &name, int m, int n, int lda, const double *mat, int precision);

/*
                 |  6   2   0   0   |
                 |  2   3   0   0   |
     A   =       |  0   0   2  -1   |
                 |  0   0  -1   2   |
*/

int main() {
    int N = 4;
    // values are saved in monotonically increasing col index
    // int ia[5] = {0, 2, 4, 6, 8};
    // int ja[8] = {0, 1, 0, 1, 2, 3, 2, 3};
    // double a[8] = {6., 2., 2., 3., 2., -1., -1., 2.};
    int ia[5] = {0, 1, 2, 3, 4};
    int ja[8] = {0, 1, 2, 3};
    double a[8] = {-8, -2, -6, -8.000001};

    sparse_matrix_t A;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, N, N, ia, ia + 1, ja, a);

    double denA[16] = {-8, 0, 0, 0,
                       0, -2, 0, 0,
                       0, 0, -6, 0,
                       0, 0, 0, -8.000001};

    double egvalues[4];

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, denA, N, egvalues);
    printmat("exact eigen veactors:", 1, N, N, egvalues, 15);
    printmat("exact eigen veactors:", N, N, N, denA, 15);

    double spE[4];
    double spEv[16];
    double Res[4];
    int k0 = 4;
    int k;
    int pm[128];
    mkl_sparse_ee_init(pm);
    pm[1] = 9;
    pm[2] = 0;
    pm[6] = 1;

    int info = mkl_sparse_d_ev("S", pm, A, descr, k0, &k, spE, spEv, Res);
    if (info != 0) {
        printf("Routine mkl_sparse_d_ev returns code of ERROR: %i", (int)info);
        return 1;
    }

    printf("#mode found/subspace %d %d \n", k, k0);
    printmat("estimated eigen values:", 1, k, k, spE, 15);
    printmat("estimated eigen vectors:", k, N, N, spEv, 15);
}

void printmat(const std::string &name, int m, int n, int lda, const double *mat, int prec) {
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