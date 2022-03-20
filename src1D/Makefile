CC=icpc -fopenmp -O3
LONGINTFLAG=-DMKL_ILP64
CFLAGS=-I"${MKLROOT}/include"
LDFLAGS_64=-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LDFLAGS_32=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

main: main.o SparseED.o
	$(CC) $^ ${LDFLAGS_64} -o $@

main.o: main.cpp SparseED.h
	$(CC) -c $<

SparseED.o: SparseED.cpp SparseED.h
	$(CC) $(LONGINTFLAG) $(CFLAGS) -c $<

.PHONY:clean
clean:
	rm -rf *.o main