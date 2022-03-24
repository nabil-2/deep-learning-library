#include "OutputAct.h"
#include "MathNN.h"

OutputAct::OutputAct(int size, v2d* downstream) : downstream(downstream) {
	this->upstream = nullptr;
	this->gradient = v2d();
}

void OutputAct::setUpstream(v2d* upstream) {
	this->upstream = upstream;
}

void OutputAct::setDownstream(v2d* downstream) {
	this->downstream = downstream;
}

v2d* OutputAct::getActivations() {
	return &activations;
}

v2d* OutputAct::getGradient() {
	return &gradient;
}

void OutputAct::forward() {
	if (this->loss == Loss::crossEntropy) {
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
	} else if (this->loss == Loss::meanSquared) {
		activations = *downstream;
	}
}

void OutputAct::setLoss(Loss loss) {
	this->loss = loss;
}

void OutputAct::backward(v2d* labels) {
	gradient = MathNN::MMsub(&activations, labels);
}