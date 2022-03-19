#include "Softmax.h"
#include "MathNN.h"

Softmax::Softmax(int size, v2d* downstream) : downstream(downstream) {
	this->upstream = nullptr;
	this->gradient = v2d();
}

void Softmax::setUpstream(v2d* upstream) {
	this->upstream = upstream;
}

void Softmax::setDownstream(v2d* downstream) {
	this->downstream = downstream;
}

v2d* Softmax::getActivations() {
	return &activations;
}

v2d* Softmax::getGradient() {
	return &gradient;
}

void Softmax::forward() {
	int width = (*downstream)[0].size();
	int height = downstream->size();
	activations = v2d(height, v1d(width));
	for (unsigned int j = 0; j < width; j++) {
		float sum = 0;
		for (unsigned int i = 0; i < height; i++) {
			sum += exp((*downstream)[i][j]);
		}
		for (unsigned int i = 0; i < height; i++) {
			activations[i][j] = exp((*downstream)[i][j]) / sum;
		}
	}
}

void Softmax::backward(v2d* labels) {
	gradient = MathNN::MMsub(&activations, labels);
}
