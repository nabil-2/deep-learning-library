#pragma once

#include <string>
#include <vector>
#include "typedef.h"
#include "Dense.h"
#include "Softmax.h"

class Network {
private:
	Fct actFct = Fct::relu;
	std::vector<Dense*> layers;
	Softmax* softmax;
public:
	Network(int inputLayerSize, int hiddenLayersCount, int hiddenLayerSize, int outputLayerSize);
	Network(int inputLayerSize, int outputLayerSize) : Network(inputLayerSize, 0, 0, outputLayerSize) {};
	void addHiddenLayer(int size);
	float train(v2d* batch, v2d* labels);
	v1d predict(v1d* input);
	void setActivationFct(Fct fct);
	void initialise();
	~Network();
};