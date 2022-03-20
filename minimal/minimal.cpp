#include <iomanip>
#include <iostream>
#include <string>

#include "mkl.h"
#include "mkl_solvers_ee.h"

void printmat(const std::string &name, int m, int n, int lda, const double *mat, int precision);

int main() {
    int N = 8;
    int ia[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int ja[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    double a[8] = {1, 2, 3, 4, 4, 3, 2, 1};

    sparse_matrix_t A;
    struct matrix_descr descr = {SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT};
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, N, N, ia, ia + 1, ja, a);

    double denA[64] = {1, 0, 0, 0, 0, 0, 0, 0,
                       0, 2, 0, 0, 0, 0, 0, 0,
                       0, 0, 3, 0, 0, 0, 0, 0,
                       0, 0, 0, 4, 0, 0, 0, 0,
                       0, 0, 0, 0, 4, 0, 0, 0,
                       0, 0, 0, 0, 0, 3, 0, 0,
                       0, 0, 0, 0, 0, 0, 2, 0,
                       0, 0, 0, 0, 0, 0, 0, 1};

    double egvalues[8];

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, denA, N, egvalues);
    printmat("exact eigen values:", 1, N, N, egvalues, 15);
    printmat("exact eigen vectors:", N, N, N, denA, 15);

    double spE[8];
    double spEv[64];
    double Res[8];
    int k0 = 5;
    int k;
    int pm[128];
    mkl_sparse_ee_init(pm);
    pm[2] = 1;
    pm[6] = 1;
    pm[7] = 1;

    int info = mkl_sparse_d_ev("S", pm, A, descr, k0, &k, spE, spEv, Res);
    if (info != 0) {
        printf("Routine mkl_sparse_d_ev returns code of ERROR: %i", (int)info);
        return 1;
    }

    switch (pm[9]) {
        case 0:
            std::cout << "The iterations stopped since convergence has been detected." << std::endl;
            break;
        case -1:
            std::cout << "Maximum number of iterations has been reached and even the \
residual norm estimates have not converged." << std::endl;
            break;
        case -2:
            std::cout << "maximum number of iterations has been reached despite the \
residual norm estimates have converged \
(but the true residuals for eigenpairs have not)." << std::endl;
            break;
        case -3:
            std::cout << "the iterations stagnated and even the residual norm estimates \
have not converged." << std::endl;
            break;
        case -4:
            std::cout << "The iterations stagnated while the eigenvalues have converged \
(but the true residuals for eigenpairs do not)." << std::endl;
            break;
        default:
            std::cout << "Unknow errors occurs." << std::endl;
    };

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