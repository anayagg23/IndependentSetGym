#include <iostream>
#include <vector>

using namespace std;

int N = 10

int independenceNumber(vector<vector<bool>> graph) {
  int n = graph.size();
  int maxSize = 0;
  for (int mask = 0; mask < (1 << n); mask++) {
    bool isIndependent = true;
    int size = 0;
    for (int i = 0; i < n; i++) {
      if (mask & (1 << i)) {
        size++;
        for (int j = 0; j < n; j++) {
          if ((mask & (1 << j)) && graph[i][j]) {
            isIndependent = false;
            break;
          }
        }
        if (!isIndependent) {
          break;
        }
      }
    }
    if (isIndependent) {
      maxSize = max(maxSize, size);
    }
  }
  return maxSize;
}

double expectedValue(double p) {
  int n = N;
  int totalGraphs = 1 << n;
  int sum = 0;

  for (int mask = 0; mask < totalGraphs; mask++) {
    vector<vector<bool>> graph(n, vector<bool>(n, false));
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        if ((mask & (1 << i)) && (mask & (1 << j)) && (rand() < p * RAND_MAX)) {
          graph[i][j] = graph[j][i] = true;
        }
      }
    }
    sum += independenceNumber(graph);
  }
  return (double)sum / totalGraphs;
}

int main() {
  double p = 0.5;  // edge probability
  cout << "Expected value of independence number: " << expectedValue(p) << endl;
  return 0;
}
