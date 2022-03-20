#ifndef SparseED2D_H
#define SparseED2D_H

typedef unsigned int uint;

class SparseED2D {
public:
    SparseED2D(int Nx, int Ny, double t, double U, double T, double mu, int Ncut);
    SparseED2D(const SparseED2D&) = delete;
    SparseED2D& operator=(const SparseED2D&) = delete;
    ~SparseED2D();

    void setT(double T);
    void setmu(double mu);

    double density() const;  // 1. density

private:
    const int Nx;    // Number of sites in x direction
    const int Ny;    // Number of sites in y direction
    const int Ns;    // Number of sites
    const int NN;    // NN = (Ns + 1) * Ns
    const int NNN;   // NNN = (Ns + 1) * NN
    const int DIM;   // DIM = 1 << Ns
    const double t;  // Hopping Energy
    const double U;  // Hopping Energy
                     //
    double T;        // Temperature
    double mu;       // Chemical potential
    int Ncut;        // Number of eigenstates under consideration

    int* Nparticle;
    int* Momentum;
    double* Ek;
    int* SubDim;
    uint** SubBase;

    int* dim;
    int* Ne;
    double** E;
    double** V;
};

#endif  // SparseED2D_H