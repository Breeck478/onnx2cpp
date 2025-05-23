#include <cmath>
#include <vector>

template<typename T>
void Tanh(const std::vector<std::vector<T>> x, const std::vector<std::vector<T>>& r){
  for (int i = 0; i <= x.size(); i++){
    
    for (int j = 0; j <= x[i].size(); j++){
      r[i][j] = sin(x[i][j]);
    }
  } 
  r = sin(x);
}




