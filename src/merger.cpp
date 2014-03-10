#include "DT.h"
#include "config.h"
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    string ifn1 = "forest1.dat", ifn2 = "forest2.dat", outfn("forest.dat");
    cout << "Usage: merger <in1 = forest1.dat> <in2 = forest2.dat> <out = forest.dat>" << endl;
    if (argc == 4) {
        ifn1 = string(argv[1]);
        ifn2 = string(argv[2]);
        outfn = string(argv[3]);
    }
    else {
        cout << "Wrong number of parameters." << endl;
    }

    ifstream ifp1(ifn1, ios::binary);
    ifstream ifp2(ifn2, ios::binary);
    if (!ifp1 || !ifp2) {
        cerr << "Cannot open the input files!" << endl;
        return -1;
    }

    auto rf = RF<labelNum, dim>::MergeRFs(ifp1, ifp2);
    ifp1.close();
    ifp2.close();

    ofstream ofp(outfn, ios::binary);
    if (!ofp) {
        cerr << "Cannot open the output file!" << endl;
        return -1;
    }
    cout << "The merged RF has " << rf.dts.size() << " trees." << endl;
    rf.Serialize(ofp);

    ofp.close();

    return 0;
}

