#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <regex>

#include "dNet.h"
#include "H5Cpp.h"
#include "nlohmann.hpp"
//#include "HDF5pp.h"

using json = nlohmann::json;
using std::cout;
using std::endl;

void readDataset(std::ifstream* stream, v2d* batch, v2d* labels, int start, int end);
void readDataset(v2d* batch, v2d* labels, std::string path, int start = 0, int end = 0);
void learnMNITS();

int main() {
    learnMNITS();
    return 0;
    const int INPUT_LAYER_SIZE = 2100;
    const int OUTPUT_LAYER_SIZE = 2;
    const int HIDDEN_LAYERS_COUNT = 2;
    const int HIDDEN_LAYERS_SIZE = 64;
    Network network(
        INPUT_LAYER_SIZE,
        HIDDEN_LAYERS_COUNT,
        HIDDEN_LAYERS_SIZE,
        OUTPUT_LAYER_SIZE
    );
    network.addHiddenLayer(64);
    network.initialise();
    network.setActivationFct(Fct::relu);

    int TEST_SIZE = 64;
    std::string dataset = "C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\dataset\\rnd.txt";
    std::ifstream filestream;
    v2d testData, testLabels;
    filestream.open(dataset);
    readDataset(&filestream, &testData, &testLabels, 0, TEST_SIZE);
    network.setTestConfig(&testData, &testLabels);
    
    int BATCHSIZE = 128,
        EPOCHS = 30;
    v2d batch, labels;
    int end;
    for (unsigned int i = 0; i < EPOCHS; i++) {
        int start = i * BATCHSIZE + TEST_SIZE;
        end = (i + 1) * BATCHSIZE + TEST_SIZE;
        readDataset(&filestream, &batch, &labels, start, end);
        float epochError = network.train(&batch, &labels);
        cout << "epoch " << i+1 << ", error=" << epochError << endl;
    }
    filestream.close();
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

void learnMNITS() {
    const int INPUT_LAYER_SIZE = 784;
    const int OUTPUT_LAYER_SIZE = 10;
    const int HIDDEN_LAYERS_COUNT = 5;
    const int HIDDEN_LAYERS_SIZE = 256;
    const int TEST_SIZE = 500;
    const int BATCHSIZE = 512;
    const int EPOCHS = 400;
    Network network(
        INPUT_LAYER_SIZE,
        HIDDEN_LAYERS_COUNT,
        HIDDEN_LAYERS_SIZE,
        OUTPUT_LAYER_SIZE
    );
    network.addHiddenLayer(128);
    network.initialise();
    network.setActivationFct(Fct::relu);

    std::string trainingDataset = "C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\dataset\\MNIST\\training";
    std::string testDataset = "C:\\Users\\nabil\\Documents\\Programmierung\\Anomaly Detection\\dataset\\MNIST\\testing";
    v2d testData = v2d(),
        testLabels = v2d();
    readDataset(&testData, &testLabels, testDataset, 1, 100);
    network.setTestConfig(&testData, &testLabels);

    v2d batch, labels;
    int end;
    for (unsigned int i = 0; i < EPOCHS; i++) {
        int start = i * BATCHSIZE + testData[0].size();
        end = (i + 1) * BATCHSIZE + testData[0].size();
        readDataset(&batch, &labels, trainingDataset, start, end);
        if (batch.size() == 0) break;
        float epochError = network.train(&batch, &labels, true);
        cout << "epoch " << i + 1 << ", error=" << epochError << endl;
    }
    cout << "finished" << endl;
    system("pause");
}

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
}

void readDataset(v2d* batch, v2d* labels, std::string path, int start, int end) {
    *batch = v2d();
    *labels = v2d();
    int i = -1;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!std::filesystem::is_directory(entry.path())) {
            i++;
            if (end > start && end != 0 && (i < start || i >= end)) continue;
            std::string imgPath{ entry.path().string() };
            int height = 0, width = 0, channels;
            v1d imgData = v1d();
            imgData.reserve(width * height);
            unsigned char* rawData = stbi_load(imgPath.c_str(), &width, &height, &channels, 1);
            for (unsigned int j = 0; j < width * height; j++) {
                imgData.push_back((float)static_cast<int>(rawData[j]) / (float)255);
            }
            stbi_image_free(rawData);
            (*batch).push_back(imgData);

            std::regex expr("\\_(\\d)\\_");
            std::smatch match;
            std::regex_search(imgPath, match, expr);
            int labelIx = stoi(match.str(1));
            v1d label = v1d(10, 0);
            label[labelIx] = 1;
            (*labels).push_back(label);
        }
    }
    if (batch->size() == 0) return;
    *batch = MathNN::transpose(batch);
    *labels = MathNN::transpose(labels);
}