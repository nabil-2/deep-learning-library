#include "MathNN.h"

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
