#pragma once

#include "typedef.h"

class Dense {
private:
	bool inputLayer;
	bool outputLayer = false;
//parameters:
	v2d weights;
	v1d bias;
	Fct activationFct;
//forward pass
	v2d activations;
	v2d* downstream = nullptr;
//backward pass
	v2d gradient;
	v2d* upstream = nullptr;
public:
	int size;
	Dense(int size, v2d* downstream);
	Dense(int size, bool inputLayer);
	void setUpstream(v2d* upstream);
	void setDownstrean(v2d* downstream);
	v2d* getActivations();
	void initialise(Fct fct);
	void forward();
	void backward();
};

