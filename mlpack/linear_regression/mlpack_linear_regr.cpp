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
    data::Load("lin_regr_in_1.txt", data);

    arma::vec responses = vectorise(data.row(1));

    // Remove last row with target values
    data.shed_row(1);
    
    cout << "Input values: \n";
    data.print(cout);
    cout << "Target values: \n";
    responses.print(cout);

    LinearRegression lr(data, responses);

    // // Get the parameters, or coefficients.
    arma::vec parameters = lr.Parameters();

    cout << "Regression line params: \n";
    parameters.print(cout);

    return 0;
}
