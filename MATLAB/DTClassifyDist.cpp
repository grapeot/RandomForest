#include "DT.h"
#include "mex.h"
#include <ppl.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

// DTClassify(Y, stage). Y is expected to be a matrix containing column vectors, and this function uses dt.dat
// to reconstruct a decision tree, and return a column-major matrix indicate the predicted distributions 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2 || nlhs > 1)
        mexErrMsgTxt("Wrong number of parameters. dist = DTClassifyDist(Y, stage);");

    const int labelNum = 10;

    // get basic parameters
#define Y_IN prhs[0]
#define STAGE_IN prhs[1]
#define ANS plhs[0]
    int m = mxGetM(Y_IN);
    int n = mxGetN(Y_IN);
    int stage = (int)mxGetScalar(STAGE_IN);
    if (m != 64) mexErrMsgTxt("Wrong dimensions.");
    double *pY = mxGetPr(Y_IN);
    ANS = mxCreateDoubleMatrix(labelNum, n, mxREAL);
    double *pAns = mxGetPr(ANS);
    stringstream ss;
    ss << "dt" << stage << ".dat";
    string ifn = ss.str();
    auto dt = DT<labelNum>::ParseFromFile(ifn);

    using namespace concurrency;
    parallel_for(0, n, [&](int i)
	{
        vector<float> x(64);
        for (int j = 0; j < 64; j++)
            x[j] = (float)pY[i * 64 + j];
        auto dist = dt->Classify(x);
        for (int j = 0; j < labelNum; j++)
            pAns[i * labelNum + j] = dist[j];
	});
    DT<10>::Dispose(dt);
    return;
}

