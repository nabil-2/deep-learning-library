#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "dNet.h"
#include "H5Cpp.h"
#include "nlohmann.hpp"
//#include "HDF5pp.h"

using json = nlohmann::json;
using std::cout;
using std::endl;

void readH5(std::string filename);
void readDataset(std::ifstream* stream, v2d* batch, v2d* labels, int start, int end);

int main() {
    const int INPUT_LAYER_SIZE = 2100;
    const int OUTPUT_LAYER_SIZE = 2;
    const int HIDDEN_LAYERS_COUNT = 1;
    const int HIDDEN_LAYERS_SIZE = 512;
    Network network(
        INPUT_LAYER_SIZE,
        HIDDEN_LAYERS_COUNT,
        HIDDEN_LAYERS_SIZE,
        OUTPUT_LAYER_SIZE
    );
    network.addHiddenLayer(256);
    network.initialise();
    
    std::string dataset = "C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\dataset\\rnd.txt";
    int BATCHSIZE = 128,
        EPOCHS = 200;
    v2d batch, labels;
    std::ifstream filestream;
    filestream.open(dataset);
    for (unsigned int i = 0; i < EPOCHS; i++) {
        int start = i * BATCHSIZE,
            end = (i + 1) * BATCHSIZE;
        readDataset(&filestream, &batch, &labels, start, end);
        float epochError = network.train(&batch, &labels);
        cout << "epoch " << i << ", error=" << epochError << endl;
    }
    filestream.close();

    /*v2d m1 = {
        {1, 2, 3},
        {4, 5, 6}
    };
    v2d m2 = {
        {7, 8, 13},
        {9, 10, 14},
        {11, 12, 15}
    };
    v2d m3 = MathNN::MMProduct(&m1, &m2);
    v1d v1 = {1, 2};
    v2d m4 = MathNN::MVadd(&m3, &v1);
    cout << exp(2) << endl;*/
}

void readDataset(std::ifstream* stream, v2d* batch, v2d* labels, int start, int end) {
    *batch = v2d();
    *labels = v2d();
    for (unsigned int i = start; i < end; i++) {
        if ((*stream).eof()) break;
        v1d eventData = v1d();
        std::string dataStr;
        getline(*stream, dataStr);
        json dataJson = json::parse(dataStr);
        v1d pT = dataJson["pT"].get<v1d>();
        v1d eta = dataJson["eta"].get<v1d>();
        v1d phi = dataJson["phi"].get<v1d>();
        eventData.insert(eventData.end(), pT.begin(), pT.end());
        eventData.insert(eventData.end(), eta.begin(), eta.end());
        eventData.insert(eventData.end(), phi.begin(), phi.end());
        int label = dataJson["signal"].get<int>();
        v1d output = (label == 1) ? v1d({ 1.f, 0.f }) : v1d({0.f, 1.f});
        batch->push_back(eventData);
        labels->push_back(output);
    }
    *batch = MathNN::transpose(batch);
    *labels = MathNN::transpose(labels);
}

void readH5(std::string filename) {
    cout << filename << endl;
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