#pragma once
#include "OnnxTensor.h"

#include <algorithm>
#include <iostream>
using namespace toCpp;
std::string OnnxTensor::GetDataTypeAsString() const {
	if (hasStaticType) {
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