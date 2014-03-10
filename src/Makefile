CC=g++
OPENMP_FLAG=-fopenmp
CFLAG=-O3 -std=c++11 -I../include -D_WHITENING

all: main merger

main: main.cpp
	$(CC) main.cpp $(CFLAG) $(OPENMP_FLAG) -o trainer
merger: merger.cpp
	$(CC) merger.cpp $(CFLAG) -o merger
clean:
	rm trainer merger
