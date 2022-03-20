CC=icpc -fopenmp -O3
LONGINTFLAG=-DMKL_ILP64
CFLAGS=-I"${MKLROOT}/include"
LDFLAGS_64=-L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
LDFLAGS_32=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

main: main.o SparseED2D.o
	$(CC) $^ ${LDFLAGS_64} -o $@

main.o: main.cpp SparseED2D.h
	$(CC) -c $<

SparseED2D.o: SparseED2D.cpp SparseED2D.h
	$(CC) $(LONGINTFLAG) $(CFLAGS) -c $<

.PHONY:clean
clean:
	rm -rf *.o main