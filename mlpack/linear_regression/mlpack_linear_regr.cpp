#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <string>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <unordered_set>
#include <set>

#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core/data/extension.hpp>

using namespace std;
using namespace mlpack::regression;
using namespace mlpack;

int main()
{
    arma::mat data; // The dataset itself.
    arma::vec responses; // The responses, one row for each row in data.

    LinearRegression lr(data, responses);

    // // Get the parameters, or coefficients.
    arma::vec parameters = lr.Parameters();
    parameters.print(cout);

    return 0;
}
