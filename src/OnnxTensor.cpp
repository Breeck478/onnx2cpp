#pragma once
#include "OnnxTensor.h"

#include <algorithm>
#include <iostream>
using namespace toCpp;
std::string OnnxTensor::GetDataTypeAsString(const bool ignorStatic) const {
	if (ignorStatic || hasStaticType) {
		return GetDataTypeString(dataType);
	}
	else {
		return "T";
	}
}

std::vector<int> OnnxTensor::Shape() const {
	return shape;
}


std::string OnnxTensor::GetVariableString() {
	std::string res = "xt::xarray<" + GetDataTypeAsString() + "> " + Name();
	if (hasStaticType) {
		res += "&";
	}
	return res;
}