#ifndef SPARSE_ED_H
#define SPARSE_ED_H

typedef unsigned int uint;

class SparseED {
public:
    SparseED(int Ns, double t, double U, double T, double mu, int Ncut);
    SparseED(const SparseED&) = delete;
    SparseED& operator=(const SparseED&) = delete;
    ~SparseED();

    void setT(double T);
    void setmu(double mu);

    double density() const;  // 1.density

private:
    const int Ns;
    const int NN;    // NN = Ns * (Ns + 1)
    const int NNN;   // NNN = (Ns + 1) * NN
    const int DIM;   // DIM = 1 << Ns
    const double t;  // Hopping Energy
    const double U;  // Onsite interaction
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

#endif  // SPARSE_ED_H