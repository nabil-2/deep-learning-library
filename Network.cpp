#include "Network.h"
#include <stdexcept>
#include <fstream>
#include <string>
#include <thread>
#include <iostream>
#include "MathNN.h"


Network::Network(int inputLayerSize, int hiddenLayersCount, int hiddenLayerSize, int outputLayerSize) {
	Dense* inputLayer = new Dense(inputLayerSize, true);
	layers.push_back(inputLayer);
	Dense* layer = inputLayer;
	for (unsigned int i = 0; i < hiddenLayersCount; i++) {
		layer = new Dense(hiddenLayerSize, layer->getActivations());
		layers.push_back(layer);
	}
	layer = new Dense(outputLayerSize, layer->getActivations());
	layers.push_back(layer);
	softmax = new Softmax(outputLayerSize, layer->getActivations());
}

void Network::evaluate() {
	v2d prediction = this->predict(testData);
	int correct = 0;
	float avgConfidence = 0.f;
	for (unsigned int j = 0; j < prediction[0].size(); j++) {
		float max = 0.f;
		int maxIx = 0,
			correctIx = 0;
		for (unsigned int i = 0; i < prediction.size(); i++) {
			if (prediction[i][j] > max) {
				max = prediction[i][j];
				maxIx = i;
			}
			if ((*testLabels)[i][j] == 1.f) {
				correctIx = i;
			}
		}
		if (maxIx == correctIx) {
			correct++;
			avgConfidence += prediction[maxIx][j];
		}
	}
	avgConfidence /= prediction[0].size();
	float accuracy = (float) correct / prediction[0].size();
	std::ofstream file(evaluationFile, std::ios_base::app);
	file << std::to_string(epoch) + ", " + std::to_string(accuracy) + ", " + std::to_string(avgConfidence) + "\n";
	std::cout << "epoch " << epoch << ", accuracy=" << accuracy << ", confidence=" << avgConfidence << std::endl;
	file.close();
}

void Network::addHiddenLayer(int size) {
	Dense* prevHiddenLayer = layers[layers.size() - 2];
	Dense* layer = new Dense(size, prevHiddenLayer->getActivations());
	auto itPos = layers.begin() + layers.size() - 1;
	layers.insert(itPos, layer);
	layers[layers.size() - 1]->setDownstrean(layer->getActivations());
}

float Network::train(v2d* batch, v2d* labels, bool saveError, std::string filename) { //batch = n(l) x batch_size
	v2d prediction = predict(batch);
	if (labels->size() != prediction.size() || (*labels)[0].size() != prediction[0].size()) {
		throw std::invalid_argument("label dimensions unequal to prediction dimensions");
	}
	v2d gradient = v2d(prediction.size(), v1d(prediction[0].size()));
	float avgError = 0;
	for (unsigned int j = 0; j < prediction[0].size(); j++) {
		float layerError = 0;
		for (unsigned int i = 0; i < prediction.size(); i++) {
			layerError += (*labels)[i][j] * log(prediction[i][j]);
			gradient[i][j] = (-1) * (*labels)[i][j] / prediction[i][j];
		}
		layerError *= -1;
		avgError += layerError;
	}
	avgError /= prediction[0].size();
	softmax->setUpstream(&gradient);
	softmax->backward(labels);
	std::vector<std::thread*> threads = std::vector<std::thread*>();
	for (unsigned int i = layers.size() - 1; i >= 1; i--) {
		threads.push_back(layers[i]->backward());
	}
	for (unsigned int i = 0; i < threads.size(); i++) {
		(*threads[i]).join();
	}
	for (unsigned int i = 0; i < threads.size(); i++) {
		delete threads[i];
	}
	epoch++;
	if (saveError) {
		std::ofstream errorFile(filename, std::ios_base::app);
		errorFile << std::to_string(epoch) + ", " + std::to_string(avgError) + "\n";
		errorFile.close();
	}
	this->evaluate();
	return avgError;
}

v1d Network::predict(v1d* input) {
	v2d twoDimensional = v2d(1, v1d(*input));
	twoDimensional = MathNN::transpose(&twoDimensional);
	v2d result = predict(&twoDimensional);
	result = MathNN::transpose(&result);
	return result[0];
}

v2d Network::predict(v2d* input) {
	layers[0]->setDownstrean(input);
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->forward();
	}
	softmax->forward();
	v2d* prediction = softmax->getActivations();
	return *prediction;
}

void Network::setTestConfig(v2d* testData, v2d* labels, std::string filename) {
	this->testData = testData;
	this->testLabels = labels;
	this->evaluationFile = filename;
}

void Network::setActivationFct(Fct fct) {
	this->actFct = fct;
}

void Network::initialise() {
	Dense* nextLayer = layers[layers.size() - 1];
	nextLayer->setUpstream(softmax->getGradient());
	for (unsigned int i = layers.size() - 2; i >= 1; i--) {
		nextLayer->initialise(actFct);
		layers[i]->setUpstream(nextLayer->getGradient());
		nextLayer = layers[i];
	}
	nextLayer->initialise(actFct);
	this->setLearningRate(this->learningRate);
}

void Network::setLearningRate(float eta) {
	this->learningRate = eta;
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->setLearningRate(eta);
	}
}

Network::~Network() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	delete softmax;
}
