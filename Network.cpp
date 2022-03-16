#include "Network.h"

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
	softmax->forward(labels);
	float error = 0;
	v2d* prediction = softmax->getActivations();
	return error;
}

v1d Network::predict(v1d* input) {
	return v1d();
}

void Network::setActivationFct(Fct fct) {
	this->actFct = fct;
}

void Network::initialise() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		layers[i]->initialise(actFct);
	}
}

Network::~Network() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	delete softmax;
}
