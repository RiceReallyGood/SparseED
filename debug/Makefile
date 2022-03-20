CC=icpc -std=c++11 -fopenmp -O3
LONGINTFLAG=-DMKL_ILP64
CFLAGS=-I"${MKLROOT}/include"
LDFLAGS_64=-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LDFLAGS_32=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

all: extendedev dimanalysis sorttime Performance OnlyE Performance2D degtest

extendedev: extendedev.cpp
	$(CC) $(CFLAGS) $^ ${LDFLAGS_32} -o $@

dimanalysis: dimanalysis.cpp
	$(CC) $(CFLAGS) $^ ${LDFLAGS_32} -o $@

sorttime: sorttime.cpp
	$(CC) $^ -o $@

Performance: Performance.cpp
	$(CC) $(LONGINTFLAG) $(CFLAGS) $^ ${LDFLAGS_64} -o $@

OnlyE: OnlyE.cpp
	$(CC) $(LONGINTFLAG) $(CFLAGS) $^ ${LDFLAGS_64} -o $@

Performance2D: Performance2D.cpp
	$(CC) $(LONGINTFLAG) $(CFLAGS) $^ ${LDFLAGS_64} -o $@

degtest: degtest.cpp
	$(CC) $(CFLAGS) $^ ${LDFLAGS_32} -o $@

.PHONY:clean
clean:
	rm -rf extendedev dimanalysis sorttime Performance OnlyE Performance2D degtest