#include "MathNN.h"
#include <stdexcept>

float MathNN::sigmoid(float x) {
	return 1 / (1 + exp(-1 * x));
}

v2d MathNN::MVProduct(v2d* matrix, v1d* vec) {
	return v2d();
}

v2d MathNN::MMProduct(v2d* matrix1, v2d* matrix2) {
	if ((*matrix1)[0].size() != matrix2->size()) {
		throw std::invalid_argument("width of matrix 1 unequal to height of matrix 2");
	}
	v2d result = v2d(matrix1->size(),  v1d((*matrix2)[0].size()));
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			v1d column = v1d(matrix2->size());
			for (unsigned int k = 0; k < matrix2->size(); k++) {
				column[k] = (*matrix2)[k][j];
			}
			result[i][j] = MathNN::VVscalar(&(matrix1->at(i)), &column);
		}
	}
	return result;
}

v2d MathNN::MMProductElementwise(v2d* matrix1, v2d* matrix2) {
	if (matrix1->size() != matrix2->size() || (*matrix1)[0].size() != (*matrix2)[0].size()) {
		throw std::invalid_argument("matrix dimensions unequal to each other");
	}
	v2d result = v2d(matrix1->size(), v1d((*matrix1)[0].size()));
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			result[i][j] = (*matrix1)[i][j] * (*matrix2)[i][j];
		}
	}
	return result;
}

v2d MathNN::MMsub(v2d* matrix1, v2d* matrix2) {
	if (matrix1->size() != matrix2->size() || (*matrix1)[0].size() != (*matrix2)[0].size()) {
		throw std::invalid_argument("matrix dimensions unequal to each other");
	}
	v2d result = v2d(matrix1->size(), v1d((*matrix1)[0].size()));
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			result[i][j] = (*matrix1)[i][j] - (*matrix2)[i][j];
		}
	}
	return result;
}

v1d MathNN::VVadd(v1d* vec1, v1d* vec2) {
	return v1d();
}

v1d MathNN::VVsub(v1d* vec1, v1d* vec2) {
	return v1d();
}

float MathNN::VVscalar(v1d* vec1, v1d* vec2) {
	if (vec1->size() != vec2->size()) {
		throw std::invalid_argument("vecotrs don't have the same dimensions");
	}
	float result = 0;
	for (unsigned int i = 0; i < vec1->size(); i++) {
		result += (*vec1)[i] * (*vec2)[i];
	}
	return result;
}

v2d MathNN::MVadd(v2d* matrix, v1d* vec) {
	if (matrix->size() != vec->size()) {
		throw std::invalid_argument("height of matrix unequal to height of vector");
	}
	auto result(*matrix);
	for (unsigned int i = 0; i < result.size(); i++) {
		for (unsigned int j = 0; j < result[i].size(); j++) {
			result[i][j] += (*vec)[i];
		}
	}
	return result;
}

v2d MathNN::activate(v2d* matrix, Fct fct) {
	auto result(*matrix);
	switch (fct) {
		case Fct::relu:
			for (unsigned int i = 0; i < result.size(); i++) {
				for (unsigned int j = 0; j < result[i].size(); j++) {
					if (result[i][j] < 0) result[i][j] = 0;
				}
			}
			break;
		case Fct::sigmoid:
			for (unsigned int i = 0; i < result.size(); i++) {
				for (unsigned int j = 0; j < result[i].size(); j++) {
					result[i][j] = 1/(1+exp(-1*result[i][j]));
				}
			}
			break;
		case Fct::tanh:
			for (unsigned int i = 0; i < result.size(); i++) {
				for (unsigned int j = 0; j < result[i].size(); j++) {
					result[i][j] = tanh(result[i][j]);
				}
			}
			break;
		case Fct::leakyRelu:
			for (unsigned int i = 0; i < result.size(); i++) {
				for (unsigned int j = 0; j < result[i].size(); j++) {
					if (result[i][j] < 0) result[i][j] *= 0.01;
				}
			}
			break;
		case Fct::swish:
			for (unsigned int i = 0; i < result.size(); i++) {
				for (unsigned int j = 0; j < result[i].size(); j++) {
					result[i][j] = result[i][j] / (1 + exp(-1 * result[i][j]));
				}
			}
			break;
		default:
			break;
	}
	return result;
}

v2d MathNN::activate_derivative(v2d* matrix, Fct fct) {
	v2d result = v2d(matrix->size(), v1d((*matrix)[0].size()));
	switch (fct) {
	case Fct::relu:
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				result[i][j] = (*matrix)[i][j] <= 0 ? 0.f : 1.f;
			}
		}
		break;
	case Fct::sigmoid:
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				float ex = exp(-1 * (*matrix)[i][j]);
				result[i][j] =  ex / pow(1 + ex, 2);
			}
		}
		break;
	case Fct::tanh:
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				result[i][j] = 1 - pow(tanh((*matrix)[i][j]), 2);
			}
		}
		break;
	case Fct::leakyRelu:
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				result[i][j] = (*matrix)[i][j] <= 0 ? 0.01f : 1.f;
			}
		}
		break;
	case Fct::swish:
		for (unsigned int i = 0; i < result.size(); i++) {
			for (unsigned int j = 0; j < result[i].size(); j++) {
				float x = (*matrix)[i][j];
				result[i][j] = sigmoid(x) + (x * sigmoid(x) * (1 - sigmoid(x)));
			}
		}
		break;
	default:
		break;
	}
	return result;
}

v2d MathNN::transpose(v2d* matrix) {
	v2d transposed((*matrix)[0].size(), std::vector<float>(matrix->size()));
	for (unsigned int i = 0; i < matrix->size(); i++) {
		for (unsigned int j = 0; j < (*matrix)[i].size(); j++) {
			transposed[j][i] = (*(matrix))[i][j];
		}
	}
	return transposed;
}

float MathNN::getNormal(float mean, float stddev, std::default_random_engine* generator) {
	std::normal_distribution<float> distribution(mean, stddev);
	return distribution(*generator);
}
