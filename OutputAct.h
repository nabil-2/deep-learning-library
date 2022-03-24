#pragma once

#include "typedef.h"

class OutputAct {
private:
	v2d activations;
	v2d gradient;
	v2d* downstream;
	v2d* upstream = nullptr;
	Loss loss = Loss::crossEntropy;
public:
	OutputAct(int size, v2d* downstream);
	void setUpstream(v2d* upstream);
	void setDownstream(v2d* downstream);
	v2d* getActivations();
	v2d* getGradient();
	void forward();
	void setLoss(Loss loss);
	void backward(v2d* labels);
};

