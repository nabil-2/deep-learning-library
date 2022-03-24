#pragma once

#include <vector>

typedef std::vector<float> v1d;
typedef std::vector<std::vector<float>> v2d;
typedef std::vector<std::vector<std::vector<float>>> v3d;

enum class Fct {
	relu,
	sigmoid,
	tanh,
	leakyRelu,
	swish,
	softmax
};

enum class Loss {
	meanSquared,
	crossEntropy
};