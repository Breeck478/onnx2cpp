#include <cmath>
#include <vector>
#include <Eigen/Dense>


template<typename T>
void Elementwise(const std::vector<std::vector<T>> x ,  std::vector<std::vector<T>>& r, T calc(T a)){
  r.resize(x.size());
for (std::size_t i = 0; i < x.size(); ++i) {
    r[i].resize(x[i].size());
}
  for (int i = 0; i < x.size(); i++){
    for (int j = 0; j < x[i].size(); j++){
      r[i][j] = calc(x[i][j]);
    }
  }
}




