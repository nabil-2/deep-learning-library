#pragma once

#include "typedef.h"
#include <thread>

class Dense {
private:
	bool inputLayer;
	bool outputLayer = false;
//parameters:
	v2d weights;
	v1d bias;
	Fct activationFct;
	std::thread* update();
//forward pass
	v2d activations;
	v2d* downstream = nullptr;
	v2d z;
//backward pass
	v2d deltaNext;
	v1d biasGradient;
	v2d weightGradient;
	v2d* upstream = nullptr;
	float learningRate;
//optimizer
	int epoch = 1;
	v1d bias_v1, bias_v2;
	v2d weight_v1, weight_v2;
	float beta1 = 0.9,
		  beta2 = 0.999;
public:
	int size;
	Dense(int size, v2d* downstream);
	Dense(int size, bool inputLayer);
	void setUpstream(v2d* upstream);
	void setDownstrean(v2d* downstream);
	void setLearningRate(float eta);
	v2d* getActivations();
	v2d* getGradient();
	void initialise(Fct fct);
	void forward();
	std::thread* backward();
};

