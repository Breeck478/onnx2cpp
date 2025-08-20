#include "OnnxTensor.h"

#include <algorithm>
#include <iostream>
using namespace toCpp;
std::string OnnxTensor::GetDataTypeAsString(const bool ignoreDynamic) const {
	if (ignoreDynamic || hasStaticType) {
		return GetDataTypeString(dataType);
	}
	else {
		return "T";
	}
}

std::vector<int> OnnxTensor::Shape() const {
	return shape;
}