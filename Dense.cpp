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
	this->deltaNext = v2d();
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

void Dense::setLearningRate(float eta) {
	this->learningRate = eta;
}

v2d* Dense::getGradient() {
	return &deltaNext;
}

void Dense::forward() {
	if (inputLayer) {
		activations = *downstream;
		return;
	}
	z = MathNN::MMProduct(&weights, downstream);
	z = MathNN::MVadd(&z, &bias);
	activations = z;
	if(!outputLayer) activations = MathNN::activate(&activations, activationFct);
}

std::thread* Dense::backward() {
	v2d gradient;
	if (outputLayer) {
		gradient = *upstream;
	} else {
		v2d activationDerivative = MathNN::activate_derivative(&z, activationFct);
		gradient = MathNN::MMProductElementwise(upstream, &activationDerivative);
	}
	v2d transposedWeights = MathNN::transpose(&weights);
	deltaNext = MathNN::MMProduct(&transposedWeights, &gradient);

	std::thread t([=]() {
		weightGradient = v2d(weights.size(), v1d(weights[0].size()));
		for (unsigned int i = 0; i < weights.size(); i++) {
			for (unsigned int j = 0; j < weights[i].size(); j++) {
				float avgWeightGradient = 0;
				for (unsigned int k = 0; k < gradient[0].size(); k++) {
					avgWeightGradient += (*downstream)[j][k] * gradient[i][k];
				}
				weightGradient[i][j] = avgWeightGradient / gradient[0].size();
			}
		}
	});	

	biasGradient = v1d(gradient.size());
	for (unsigned int i = 0; i < gradient.size(); i++) {
		float avgBiasGradient = 0.f;
		for (unsigned int j = 0; j < gradient[i].size(); j++) {
			avgBiasGradient += gradient[i][j];
		}
		biasGradient[i] = avgBiasGradient / gradient[i].size();
	}

	t.join();
	return this->update();
}

std::thread* Dense::update() {
	std::thread* t = new std::thread([=]() {
		float eta = learningRate,
			offset = 0.00001;
		for (unsigned int i = 0; i < bias.size(); i++) {
			bias_v1[i] = beta1 * bias_v1[i] + (1 - beta1) * biasGradient[i];
			bias_v2[i] = beta2 * bias_v2[i] + (1 - beta2) * pow(biasGradient[i], 2);
			float bias_v1h = bias_v1[i] / (1 - pow(beta1, epoch));
			float bias_v2h = bias_v2[i] / (1 - pow(beta2, epoch));
			bias[i] -= eta * bias_v1h / sqrt(bias_v2h + offset);
			for (unsigned int j = 0; j < weights[i].size(); j++) {
				weight_v1[i][j] = beta1 * weight_v1[i][j] + (1 - beta1) * weightGradient[i][j];
				weight_v2[i][j] = beta2 * weight_v2[i][j] + (1 - beta2) * pow(weightGradient[i][j], 2);
				float weight_v1h = weight_v1[i][j] / (1 - pow(beta1, epoch));
				float weight_v2h = weight_v2[i][j] / (1 - pow(beta2, epoch));
				weights[i][j] -= eta * weight_v1h / sqrt(weight_v2h + offset);
			}
		}
		epoch++;
	});
	return t;
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
	bias_v1 = v1d(size, 0.f);
	bias_v2 = v1d(size, 0.f);
	weight_v1 = v2d(size, v1d(downstream->size(), 0.f));
	weight_v2 = v2d(size, v1d(downstream->size(), 0.f));
}
