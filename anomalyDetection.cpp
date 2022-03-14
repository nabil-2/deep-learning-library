#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "dNet.h"
#include "H5Cpp.h"
//#include "HDF5pp.h"

using std::cout;
using std::endl;

void readH5(std::string filename);
void readDataset(std::string filename);

int main() {
    
    //readH5("C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\dataset\\rnd_tiny.h5");
    readH5("C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\src\\dNet\\h5test2.h5");
}

void readH5(std::string filename) {
    /*int data[2][2] = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<int>> data2 = {
        {1, 2},
        {3, 4}
    };
    std::ofstream file("my_file.txt");
    for (const auto &i : data2) {
        for (const auto &j : i) {
            file << j << endl;
        }
    }
    H5::H5File file("h5test.h5", H5F_ACC_TRUNC);
    hsize_t dimsf[2];
    dimsf[0] = 2;
    dimsf[1] = 2;
    H5::DataSpace dataspace(2, dimsf);
    H5::IntType datatype(H5::PredType::NATIVE_INT);
    datatype.setOrder(H5T_ORDER_LE);
    H5::DataSet dataset = file.createDataSet("ds", datatype, dataspace);
    dataset.write(data, H5::PredType::NATIVE_INT);

    H5::H5File file2(filename, H5F_ACC_TRUNC);
    H5::DataSet dataset2 = file.openDataSet("ds");
    H5::DataSpace filespace = dataset2.getSpace();
    std::cout << filespace.getSimpleExtentNdims() << std::endl;*/
}

void readDataset(std::string filename) {    
    std::ifstream inFile;
	inFile.open(filename);
	std::string data;
	while (!inFile.eof()) {
		std::string tmp;
		getline(inFile, tmp);

	}
	inFile.close();
}