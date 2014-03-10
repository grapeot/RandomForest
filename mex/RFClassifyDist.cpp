#include "../include/DT.h"
#include "../include/config.h"
#include "mex.h"
#include <string>
#include <iostream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

RF<labelNum, dim> *prf = NULL;

// RFClassify(Y). Y is expected to be a matrix containing column vectors, and this function uses tree.dat
// to reconstruct a decision tree, and return a column-major matrix indicate the predicted distributions 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1 || nlhs > 1)
        mexErrMsgTxt("Wrong number of parameters. dist = RFClassifyDist(Y);");  

    // get basic parameters
#define Y_IN prhs[0]
#define ANS plhs[0]
    int m = mxGetM(Y_IN);
    int n = mxGetN(Y_IN);

    if (!m && !n) {
        mexPrintf("Memory sweeping...\n");
        if (!prf) {
            mexPrintf("Memory not using... Do nothing.\n");
            return;
        }
        prf->Dispose();
        delete prf;
        prf = NULL;
        return;
    }

    if (m != dim) mexErrMsgTxt("Wrong dimensions.");
    double *pY = mxGetPr(Y_IN);
    ANS = mxCreateDoubleMatrix(labelNum, n, mxREAL);
    double *pAns = mxGetPr(ANS);
    string ifn = "forest.dat";
    ifstream ifp(ifn, ios::binary);
    if (!ifp) {
        mexErrMsgTxt("Cannot read in the random forest data file!");
    }
    if (!prf) {
        mexPrintf("Loading random forest...\n");
        prf = new RF<labelNum, dim>();
        auto rf = RF<labelNum, dim>::ParseFromStream(ifp);
        for (auto dt: rf.dts) {
            prf->dts.push_back(dt);
        }
    }
    ifp.close();

    for( int i = 0; i < n; i++)
    {
        vector<float> x(dim);
        for (int j = 0; j < dim; j++)
        {
            x[j] = (float)pY[i * dim + j];
        }

        auto dist = prf->Classify(x);
        for (int j = 0; j < labelNum; j++)
            pAns[i * labelNum + j] = dist[j];
    }

    return;
}

