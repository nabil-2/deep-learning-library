#pragma once

#include "typedef.h"

class Softmax {
private:
	v2d activations;
	v2d gradient;
	v2d* downstream;
	v2d* upstream = nullptr;
public:
	Softmax(int size, v2d* downstream);
	void setUpstream(v2d* upstream);
	void setDownstream(v2d* downstream);
	v2d* getActivations();
	void forward(v2d* labels);
	void backward();
};

