MATLAB_PATH=/home/grapeot/MATLAB/R2014a
CC=g++
OPENMP_FLAG=-fopenmp
CXX_FLAG=-std=c++11 -O3

all: dt rf 

dt: DTClassifyDist.cpp
	$(CC) -c -DMX_COMPAT_32   -D_GNU_SOURCE -DMATLAB_MEX_FILE  -I"$(MATLAB_PATH)/extern/include" -I"$(MATLAB_PATH)/simulink/include" -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -DNDEBUG DTClassifyDist.cpp -o DTClassifyDist.o $(CXX_FLAG) $(OPENMP_FLAG); \
	$(CC) -pthread -Wl,--no-undefined  -shared -O -Wl,--version-script,"$(MATLAB_PATH)/extern/lib/glnxa64/mexFunction.map" DTClassifyDist.o   -Wl,-rpath-link,$(MATLAB_PATH)/bin/glnxa64 -L"$(MATLAB_PATH)/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -o DTClassifyDist.mexa64 $(OPENMP_FLAG); \
	rm DTClassifyDist.o

rf: RFClassifyDist.cpp
	$(CC) -c -DMX_COMPAT_32   -D_GNU_SOURCE -DMATLAB_MEX_FILE  -I"$(MATLAB_PATH)/extern/include" -I"$(MATLAB_PATH)/simulink/include" -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -DNDEBUG RFClassifyDist.cpp -o RFClassifyDist.o $(CXX_FLAG) $(OPENMP_FLAG); \
	$(CC) -pthread -Wl,--no-undefined  -shared -O -Wl,--version-script,"$(MATLAB_PATH)/extern/lib/glnxa64/mexFunction.map" RFClassifyDist.o   -Wl,-rpath-link,$(MATLAB_PATH)/bin/glnxa64 -L"$(MATLAB_PATH)/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -o RFClassifyDist.mexa64 $(OPENMP_FLAG); \
	rm RFClassifyDist.o
