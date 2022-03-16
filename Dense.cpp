#include <random>
#include "Dense.h"
#include "MathNN.h"

Dense::Dense(int size, v2d* downstream) : Dense(size, false) { //not input Layer
	this->downstream = downstream;
}

Dense::Dense(int size, bool inputLayer) {
	this->size = size;
	this->inputLayer = inputLayer;
	this->activations = v2d(size, v1d()); //h=n(l); w=batch_size
}

void Dense::setUpstream(v2d* upstream) {
	this->upstream = upstream;
}

void Dense::setDownstrean(v2d* downstream) {
	this->downstream = downstream;
}

v2d* Dense::getActivations() {
	return &activations;
}

void Dense::forward() {
	if (inputLayer) {
		activations = *downstream;
		return;
	}
	v2d z = MathNN::MMProduct(&weights, downstream);
	activations = MathNN::MVadd(&z, &bias);
	if(!outputLayer) activations = MathNN::activate(&activations, activationFct);
}

void Dense::backward() {
}

void Dense::initialise(Fct fct) {
	this->activationFct = fct;
	if(this->inputLayer) return;
	float mean = 0;
	float stddev = 0;
	switch (fct) {
		case Fct::softmax:
			this->outputLayer = true;
			return;
		case Fct::relu:
		case Fct::leakyRelu:
		case Fct::swish:
			stddev = sqrt(2.f / downstream->size()); //he
			break;
		case Fct::sigmoid:
		case Fct::tanh:
			stddev = sqrt(2.f / (downstream->size() + size)); //xavier
			break;
		default:
			break;
	}
	bias = v1d(size, 0.f); //h=n(l)
	weights = v2d(size, v1d(downstream->size())); //h=n(l); w=n(l-1)
	std::default_random_engine generator;
	for (unsigned int i = 0; i < weights.size(); i++) {
		for (unsigned int j = 0; j < weights[i].size(); j++) {
			weights[i][j] = MathNN::getNormal(mean, stddev, &generator);
		}
	}
}
