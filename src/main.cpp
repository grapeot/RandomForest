#include "DT.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstring>
#include <cstdlib>

using namespace std;

int main(int argc, char **argv)
{
	int level = 6, treeNum = 6;

	string ifn, ofn;
	if (argc == 1) 
	{
		ifn = "in.dat";
#ifdef USE_DT
		ofn = "tree.dat";
        cout << "Usage: dttest <in file> <out file> <level>" << endl
            << "Train a decision tree. No parameters detected. Use default parameters." << endl;
        cout << "Input filename: in.dat" << endl
            << "Output filename: tree.dat" << endl;
#else
        ofn = "forest.dat";
        cout << "Usage: dttest <in file> <out file> <level> <tree num>" << endl
            << "Train a random forest. No parameters detected. Use default parameters." << endl;
        cout << "Input filename: in.dat" << endl
            << "Output filename: forest.dat" << endl;
#endif
	}
	else
	{
		ifn = argv[1];
		ofn = argv[2];
		if (argc > 3)
			level = atoi(argv[3]);
#ifndef USE_DT
        if (argc > 4)
            treeNum = atoi(argv[4]);
#endif

	}

	cout << "Label #: " << labelNum << endl
        << "Level: " << level << endl;
#ifndef USE_DT
	cout << "Tree #: " << treeNum << endl;
#endif

	ifstream ifp(ifn, ios::binary);
	if (!ifp) throw exception();
	int count = 0;
	ifp.read((char *)&count, sizeof(int));
	vector<float> x(count * dim);
	vector<int> labels(count);
	ifp.read((char *)&labels[0], sizeof(int) * count);
	ifp.read((char *)&x[0], sizeof(float) * count * dim);
	ifp.close();
	cout << count << " elements read in." << endl;
	auto cc = clock();
#ifdef USE_DT
	auto dt = DT<labelNum, dim>::Train(x, labels, level);
#else
    auto rf = RF<labelNum, dim>::Train(x, labels, treeNum, level);
#endif
	cout << endl << clock() - cc << " ms." << endl;

	ofstream ofp(ofn, ios::binary);
	if (!ofp) throw exception();
#ifdef USE_DT
	dt->Serialize(ofp);
#else
    rf.Serialize(ofp);
#endif
	ofp.close();

	ifp.open(ofn, ios::binary);
	if (!ifp) throw exception();
#ifdef USE_DT
	DT<labelNum, dim>::Dispose(dt);
	dt = DT<labelNum, dim>::ParseFromStream(ifp);
#else
    rf.Dispose();
    rf = RF<labelNum, dim>::ParseFromStream(ifp);
#endif

	ifp.close();

	cout << "Classifier read in memory..." << endl;
	vector<float> x2(dim);
	float c = 0;
	int n = labels.size();
	for(int i = 0; i < n; i++)
	{
		memcpy(&x2[0], &x[i * dim], sizeof(float) * dim);
#ifdef USE_DT
		auto predictedY = dt->ClassifyLabel(x2);
#else
        auto predictedY = rf.ClassifyLabel(x2);
#endif
		if (labels[i] == predictedY) c++;
	}
	cout << "Average prediction accuracy: " << c / n << endl;

#ifdef USE_DT
	DT<labelNum, dim>::Dispose(dt);
#else
    rf.Dispose();
#endif

	return 0;
}
