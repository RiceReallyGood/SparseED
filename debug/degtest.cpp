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
    int N = 8;
    // values are saved in monotonically increasing col index
    // int ia[5] = {0, 2, 4, 6, 8};
    // int ja[8] = {0, 1, 0, 1, 2, 3, 2, 3};
    // double a[8] = {6., 2., 2., 3., 2., -1., -1., 2.};
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
    /* double q[64] = {-0.683485, -0.393213, -0.291538, 0.523952, 0.0497908, 0.0935999, 0.00261843, -0.0864031, 
                  0.356366, -0.449531,  0.392889, 0.266801, 0.251665,  0.529342, -0.220424,    0.230774, 
                  0.310227,  0.49523,  -0.449389, 0.413905, 0.0757211, 0.312408, -0.294613,   -0.308396, 
                  0.528751, -0.470202, -0.395097, 0.151908, -0.269213, -0.171647, 0.452541, -0.115859, 
                  0.143749, 0.0707206, 0.239534, 0.542721, -0.0823601, -0.680535, -0.315853, 0.229647,
                  0.0637636, -0.215151, -0.565723, -0.325838, 0.37475, -0.177086, -0.419522, 0.419089,
                  -0.0719997, 0.0868353, -0.135986, 0.00368107, -0.764151, 0.290801, -0.155363, 0.525491,
                  -0.0109606, 0.343701, -0.0817361, 0.243245, 0.353312, 0.0701582, 0.59993, 0.571168};
    double temp[64];
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, q, N, denA, N, 0, temp, N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, temp, N, q, N, 0, denA, N); */

    double egvalues[8];

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, denA, N, egvalues);
    printmat("exact eigen veactors:", 1, N, N, egvalues, 15);
    printmat("exact eigen veactors:", N, N, N, denA, 15);

    double spE[8];
    double spEv[64];
    double Res[8];
    int k0 = 5;
    int k;
    int pm[128];
    mkl_sparse_ee_init(pm);
    pm[1] = 9;
    pm[2] = 0;
    pm[3] = 8;
    pm[6] = 1;
    pm[7] = 1;

    int info = mkl_sparse_d_ev("S", pm, A, descr, k0, &k, spE, spEv, Res);
    if (info != 0) {
        printf("Routine mkl_sparse_d_ev returns code of ERROR: %i", (int)info);
        return 1;
    }

    // switch (pm[9]) {
    //     case 0:
    //         std::cout << "The iterations stopped since convergence has been detected." << std::endl;
    //         break;
    //     case -1:
    //         std::cout << "Maximum number of iterations has been reached and even the \
    //                       residual norm estimates have not converged." << std::endl;
    //         break;
    //     case -2:
    //         std::cout << "maximum number of iterations has been reached despite the \
    //                       residual norm estimates have converged \
    //                       (but the true residuals for eigenpairs have not)." << std::endl;
    //         break;
    //     case -3:
    //         std::cout << "the iterations stagnated and even the residual norm estimates \
    //                       have not converged." << std::endl;
    //         break;
    //     case -4:
    //         std::cout << "The iterations stagnated while the eigenvalues have converged \
    //                       (but the true residuals for eigenpairs do not)." << std::endl;
    //         break;
    //     default:
    //         std::cout << "Unknow errors occurs." << std::endl;
    // };

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