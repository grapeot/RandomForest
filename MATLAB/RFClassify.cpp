#include "../DT.h"
#include "mex.h"
#include <ppl.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

// RFClassify(Y, stage). Y is expected to be a matrix containing column vectors, and this function uses dt.dat
// to reconstruct a decision tree, and return a row vector indicate the predicted labels
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2 || nlhs > 1)
        mexErrMsgTxt("Wrong number of parameters. ret = DTClassify(Y, stage);");

    // get basic parameters
#define Y_IN prhs[0]
#define STAGE_IN prhs[1]
#define ANS plhs[0]
    int m = mxGetM(Y_IN);
    int n = mxGetN(Y_IN);
    int stage = (int)mxGetScalar(STAGE_IN);
    if (m != 64) mexErrMsgTxt("Wrong dimensions.");
    double *pY = mxGetPr(Y_IN);
    ANS = mxCreateDoubleMatrix(1, n, mxREAL);
    double *pAns = mxGetPr(ANS);
    stringstream ss;
    ss << "rf" << stage << ".dat";
    string ifn = ss.str();
    ifstream ifp(ifn, ios::binary);
    if (!ifp) throw exception();
    auto rf = RF<10>::ParseFromStream(ifp);
    ifp.close();

    for(int i = 0; i < n; i++)
	{
        vector<float> x(64);
        for (int j = 0; j < 64; j++)
            x[j] = (float)pY[i * 64 + j];
        auto xx = rf.ClassifyLabel(x);
        pAns[i] = xx[0] + 1;
	}
    rf.Dispose();
    return;
}
