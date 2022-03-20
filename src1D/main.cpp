#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "SparseED.h"
using namespace std;

int main() {
    int Ns = 8;
    double t = 1., U = 4.;
    int Ncut = 100;
    SparseED mysystem(Ns, t, U, 1., U / 2, Ncut);

    double dmu = 0.025, eps = 1e-8;
    int nmu = int((U / 2 - (-U) + eps) / dmu) + 1;
    vector<double> mutab(nmu);
    for (int i = 0; i < nmu; i++) {
        mutab[i] = -U + i * dmu;
    }

    vector<double> betatab = {1, 2, 4, 6, 8, 16, 32};

    for (int bid = 0; bid < betatab.size(); bid++) {
        ostringstream dirname("data/", ios::ate);
        dirname << "Ns" << Ns << "/beta" << betatab[bid] << "/U" << U << "/";
        int status = system(("mkdir -p " + dirname.str()).c_str());
        if (status) {
            cerr << "Can not make diretory \"" << dirname.str() << "\"" << std::endl;
            exit(status);
        }

        ofstream denfile((dirname.str() + "DenVSmu.dat").c_str());
        if (!denfile) {
            cerr << "Can open file \"" << dirname.str() << "DenVSmu.dat"
                 << "\"" << std::endl;
            exit(1);
        }
        denfile.setf(ios::fixed);

        mysystem.setT(1. / betatab[bid]);
        for (int muid = 0; muid < nmu; muid++) {
            mysystem.setmu(mutab[muid]);
            denfile << setprecision(3) << setw(5) << mutab[muid] << "\t"
                    << setprecision(8) << setw(10) << mysystem.density() << endl;
        }

        denfile.close();
    }
}