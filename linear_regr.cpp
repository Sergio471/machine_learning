#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

using namespace std;

vector<pair<double, double>> get_points() {

    vector<pair<double, double>> result;
    
    double a, b;
    for (int i = 0; i < 1036; ++i) {
        cin >> a >> b;
        result.push_back({a, b});
    }
    
    return result;
}

vector<pair<double, double>> get_random_points(int k, int b, int s, int e, int disp) {
    
    vector<pair<double, double>> result;
    result.reserve(e - s + 1);

    for (int x = s; x <= e; ++x) {
        double ideal_y = k * x + b;
        double y = ideal_y + (-disp + 2 * (rand() % (disp + 1)));
        result.push_back({x, y});
    }

    return result;
}

// gradient descent
pair<double, double> get_kb(const vector<pair<double, double>>& points) {
    double k = 100.0f, b = 100.0f;

    double alpha_k = 0.000001;
    double alpha_b = 0.0005;
    int N = points.size();
    double err = 1000000.0f;
    double dk = 0.0f;// -- dE/dk
    double db = 0.0f;// -- dE/db
    while (err > 15.0) {
        dk = 0.0f;
        db = 0.0f;
        for (auto& p : points) {
            dk += -p.first * (p.second - (k * p.first + b));
            db += -(p.second - (k * p.first + b));
        }
        dk /= N;
        dk *= 2;
        db /= N;
        db *= 2;
        
        k = k - alpha_k * dk;
        b = b - alpha_b * db;
        
        err = 0.0f;
        for (auto& p : points) {
            err += pow(p.second - (p.first * k + b), 2);
        }
        err /= N;
    }
    cout << "Simple grad desc error: " << err << endl;
    
    return {k, b};
}

// stochastic gd
vector<double> get_ks(const vector<pair<double, double>>& points) {
    double k0 = 10.0, k1 = 10.0; // k0 - b, k1 - k
    
    double alpha_k0 = 0.0001;
    double alpha_k1 = 0.00001;
    int N = points.size();
    double err = 0.0f;//1000000.0f;
    double dk0 = 0.0f;
    double dk1 = 0.0f;
    int batch_size = 1;
    for (int i = 0; i < points.size() - batch_size; i += batch_size) {
        
        dk0 = dk1 = 0.0f;
        
        for (int j = 0; j < batch_size; ++j) {

            double skob = (points[i + j].second - (k0 + k1*points[i + j].first));

            dk0 += -2 * skob;
            dk1 += -2 * points[i + j].first * skob; 
            
        }
        dk0 /= (batch_size / 2);
        dk1 /= (batch_size / 2);      

        cout << "Derivatives: " << dk0 << ", " << dk1 << endl;
        
        k0 = k0 - alpha_k0 * dk0;
        k1 = k1 - alpha_k1 * dk1;                   
    }
                      
    err = 0.0f;
    for (int i = 0; i < points.size(); ++i) {
        
        double skob = (points[i].second - (k0 + k1*points[i].first));
        skob *= skob;
        err += skob;
    }
    err /= points.size();    
    cout << "Stochastic err: " << err << endl;
    
    return {k0, k1};
}

int main() {
    srand(time(0));

    cout << "Stachastic with many params" << endl;
    const vector<pair<double, double>> points = get_points();
    vector<double> ks = get_ks(points);
    cout << ks[0] << " + " << ks[1] << "*x" << endl;

    cout << "\n-------------------\n";

    cout << "Simple with two params" << endl;
    pair<double, double> kb = get_kb(points);
    cout << kb.first <<  "*x + " << kb.second << endl;
    cout << endl;
    
    return 0;
}
