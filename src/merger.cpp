#include "DT.h"
#include "config.h"
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Wrong number of parameters." << endl;
        cout << "Usage: merger <out = forest.dat> <list of forest files>" << endl;
        exit(-1);
    }

    // first random forest with data structure
    string ifn1 = argv[2];
    ifstream ifp1(ifn1, ios::binary);
    if (!ifp1) {
        cerr << "Cannot open the input files!" << endl;
        return -1;
    }
    RF<labelNum, dim> rf;
    int count = 0, dti = 0;
    ifp1.read((char *)&count, sizeof(int));
    rf.dts.reserve(count * (argc - 2));
    cout << "Merging random forest " << 1 << " from " << ifn1 << ", with " << count << " trees..." << endl;
    for(; dti < count; dti++)
        rf.dts.push_back(DT<labelNum, dim>::ParseFromStream(ifp1));

    // remaining ones
    ifstream ifp2;
    for (int i = 3; i < argc; i++) {
        ifp2.open(argv[i], ios::binary);
        if (!ifp2) {
            cerr << "Cannot open the input files!" << endl;
            return -1;
        }
        ifp2.read((char *)&count, sizeof(int));
        cout << "Merging random forest " << i - 1 << " from " << argv[i] << ", with " << count << " trees..." << endl;
        for(int j = 0; j < count; j++)
            rf.dts.push_back(DT<labelNum, dim>::ParseFromStream(ifp2));
        ifp2.close();
        ifp2.clear();
    }

    string outfn = argv[1];
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

