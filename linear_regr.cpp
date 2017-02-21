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
pair<double, double> get_kb(vector<pair<double, double>>& points) {
    double k = 0.0f, b = 0.0f;

    {
        double err = 0.0f;
        for (auto& p : points) {
            err += pow(p.second - (p.first * k + b), 2);
        }
        err /= points.size();
        cout << "before err: " << err << endl;
    }
    
    double alpha_k = 0.000000001;
    double alpha_b = 0.000005;
    int N = points.size();
    double err = 1000000.0f;
    double dk = 0.0f;// -- dE/dk
    double db = 0.0f;// -- dE/db
    while (err > 0.05) {
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
        
        double new_k = k - alpha_k * dk;
        double new_b = b - alpha_b * db;
        
        float new_err = 0.0f;
        for (auto& p : points) {
            new_err += pow(p.second - (p.first * new_k + new_b), 2);
        }
        new_err /= N;

        if (new_err < err - 0.001) {
            k = new_k;
            b = new_b;
            err = new_err;
        } else {
            break;
        }
    }
    cout << "after err: " << err << endl;
    
    return {k, b};
}

// stochastic gd
vector<double> get_ks(vector<pair<double, double>>& points) {
    double k0 = 10.0, k1 = 10.0, k2 = 10.0, k3 = 10.0;
    
    double alpha_k0 = 0.0001;
    double alpha_k1 = 0.00001;
    double alpha_k2 = 0.0000001;
    double alpha_k3 = 0.0000000001;
    int N = points.size();
    double err = 0.0f;//1000000.0f;
    double dk0 = 0.0f;
    double dk1 = 0.0f;
    double dk2 = 0.0f;
    double dk3 = 0.0f;
    int batch_size = 1000;
    for (int i = 0; i < points.size() - batch_size; i += batch_size) {
        
        dk0 = dk1 = dk2 = dk3 = 0.0f;
        
        for (int j = 0; j < batch_size; ++j) {

            double skob = (points[i + j].second - (k0 + 
                                              k1*points[i + j].first +
                                              k2*pow(points[i + j].first, 2) +
                                              k3*pow(points[i + j].first, 3)));

            dk0 += -2 * skob;
            dk1 += -2 * points[i + j].first * skob;
            dk2 += -2 * pow(points[i + j].first, 2) * skob;
            dk3 += -2 * pow(points[i + j].first, 3) * skob;  
            
        }
        dk0 /= (batch_size / 2);
        dk1 /= (batch_size / 2);
        dk2 /= (batch_size / 2);
        dk3 /= (batch_size / 2);        
        
        cout << "dks: " << dk0 << " " << dk1 << " " << dk2 << " " << dk3 << endl;
        
        k0 = k0 - alpha_k0 * dk0;
        k1 = k1 - alpha_k1 * dk1;
        k2 = k2 - alpha_k2 * dk2;
        k3 = k3 - alpha_k3 * dk3;                      
        
        cout << "ks: " << k0 << " " << k1 << " " << k2 << " " << k3 << endl;
        
    }
                      
    err = 0.0f;
    for (int i = 0; i < points.size(); ++i) {
        
        double skob = (points[i].second - (k0 + 
                                          k1*points[i].first +
                                          k2*pow(points[i].first, 2) +
                                          k3*pow(points[i].first, 3)));
        skob *= skob;
        err += skob;
    }
    err /= points.size();    
    cout << "Err: " << err << endl;
    
    return {k0, k1, k2, k3};
}

int main() {
    srand(time(0));

    if (false)
    {
        // generated input
        int k = 2, b = 10;
        vector<pair<double, double>> points = get_random_points(k, b, 0, 100, 0);

        pair<double, double> kb = get_kb(points);
        cout << kb.first <<  "*x +  " << kb.second << endl;
    }
    
    {
        // input from source
        vector<pair<double, double>> points = get_points();
        vector<double> ks = get_ks(points);
        for (auto k : ks) cout << k << " ";
        cout << "\n-------------------\n";
        pair<double, double> kb = get_kb(points);
        cout << kb.first <<  "*x +  " << kb.second << endl;
        cout << endl;
    }
    
    return 0;
}
