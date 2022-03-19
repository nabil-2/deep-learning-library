#include "Network.h"
#include <stdexcept>

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

void Network::addHiddenLayer(int size) {
	Dense* prevHiddenLayer = layers[layers.size() - 2];
	Dense* layer = new Dense(size, prevHiddenLayer->getActivations());
	auto itPos = layers.begin() + layers.size() - 1;
	layers.insert(itPos, layer);
	layers[layers.size() - 1]->setDownstrean(layer->getActivations());
}

float Network::train(v2d* batch, v2d* labels) { //batch = n(l) x batch_size
	layers[0]->setDownstrean(batch);
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->forward();
	}
	softmax->forward();
	v2d* prediction = softmax->getActivations();
	if (labels->size() != prediction->size() || (*labels)[0].size() != (*prediction)[0].size()) {
		throw std::invalid_argument("label dimensions unequal to prediction dimensions");
	}
	v2d gradient = v2d(prediction->size(), v1d((*prediction)[0].size()));
	float avgError = 0;
	for (unsigned int j = 0; j < (*prediction)[0].size(); j++) {
		float layerError = 0;
		for (unsigned int i = 0; i < prediction->size(); i++) {
			layerError += (*labels)[i][j] * log((*prediction)[i][j]);
			gradient[i][j] = (-1) * (*labels)[i][j] / (*prediction)[i][j];
		}
		layerError *= -1;
		avgError += layerError;
	}
	avgError /= (*prediction)[0].size();
	softmax->setUpstream(&gradient);
	softmax->backward(labels);
	for (unsigned int i = layers.size() - 1; i >= 1; i--) {
		layers[i]->backward();
	}
	return avgError;
}

v1d Network::predict(v1d* input) {
	return v1d();
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
