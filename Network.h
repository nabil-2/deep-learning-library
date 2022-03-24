#pragma once

#include <string>
#include <vector>
#include "typedef.h"
#include "Dense.h"
#include "OutputAct.h"

class Network {
private:
	Fct actFct = Fct::relu;
	std::vector<Dense*> layers;
	OutputAct* outAct;
	float learningRate = 0.0005;
	int epoch = 0;
	v2d *testData = nullptr,
		*testLabels = nullptr;
	std::string evaluationFile;
	void evaluate();
	Loss loss = Loss::crossEntropy;
public:
	Network(int inputLayerSize, int hiddenLayersCount, int hiddenLayerSize, int outputLayerSize);
	Network(int inputLayerSize, int outputLayerSize) : Network(inputLayerSize, 0, 0, outputLayerSize) {};
	void addHiddenLayer(int size);
	float train(v2d* batch, v2d* labels, bool saveError = false, std::string filename = "error_epoch.csv");
	v1d predict(v1d* input);
	v2d predict(v2d* input);
	void setTestConfig(v2d* testData, v2d* labels, std::string targetFile = "evaluation.csv");
	void setActivationFct(Fct fct);
	void initialise();
	void setLearningRate(float eta);
	void setLoss(Loss loss);
	~Network();
};