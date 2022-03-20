CC=icpc -std=c++11 -O3
CFLAGS=-I"${MKLROOT}/include"
LDFLAGS=-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

minimal: minimal.cpp
	$(CC) $(CFLAGS) $^ ${LDFLAGS} -o $@

.PHONY:clean
clean:
	rm minimal