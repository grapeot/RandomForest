if ~ispc
    error('This file is only used to compile the mex files on Windows. Use GNU make for Linux and Mac.');
end

mex DTClassifyDist.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex RFClassifyDist.cpp COMPFLAGS="/openmp $COMPFLAGS"