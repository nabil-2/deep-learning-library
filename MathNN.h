#pragma once

#include <vector>
#include <random>
#include "typedef.h"

class MathNN {
public:
	v2d static MVProduct(v2d* matrix, v1d* vec);
	v2d static MMProduct(v2d* matrix1, v2d* matrix12);
	v1d static VVadd(v1d* vec1, v1d* vec2);
	v1d static VVsub(v1d* vec1, v1d* vec2);
	float static VVscalar(v1d* vec1, v1d* vec2);
	v2d static MVadd(v2d* matrix, v1d* vec);
	v2d static activate(v2d* matrix, Fct fct);
	v2d static transpose(v2d* matrix);
	float static getNormal(float mean, float stddev, std::default_random_engine* generator);
};

