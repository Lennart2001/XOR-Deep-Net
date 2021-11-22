//
//  deep network.cpp
//
//  Created by Lennart Buhl on 11/20/21.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
    


#define INPUT 2
#define Z1 4
#define Z2 4
#define OUTPUT 2
#define EPOCH 10000

#define LEARN_RATE 0.01


using namespace std;


double relu(double in) {
    return fmax(0.0, in);
}

double relu_derivative(double in) {
    if (in > 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}


vector<double> softmax(vector<double> in) {
    
    double m = -INFINITY;
    for (int i = 0; i < in.size(); i++) {
        if (in[i] > m) {
            m = in[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < in.size(); i++) {
        sum += exp(in[i] - m);
    }
    double offset = m + log(sum);
    for (int i = 0; i < in.size(); i++) {
        in[i] = exp(in[i] - offset);
    }
    return in;
}

vector<double> softmax_derivative(vector<double> in) {
   
      vector<double> act = softmax(in);
    
      for (int i = 0; i < act.size(); i++) {
        in[i] = act[i] * (1.0 - act[i]);
      }
    return in;
}


int main() {
    
    srand((int)time(0));
    
    vector<vector<double>> weight1(INPUT, vector<double>(Z1, 0));
    vector<vector<double>> weight2(Z1, vector<double>(Z2, 0));
    vector<vector<double>> weight3(Z2, vector<double>(OUTPUT, 0));
    vector<double> b1(Z1, 0), b2(Z2, 0), b3(OUTPUT, 0);
    vector<double> error_vec;
    
    
    //MARK: INIT WEIGHTS
    for (int x = 0; x < INPUT; x++) {
        for (int y = 0; y < Z1; y++) {
            weight1[x][y] = rand() / (double)(RAND_MAX);
        }
        b1[x] = rand() / (double)(RAND_MAX);
    }
    for (int x = 0; x < Z1; x++) {
        for (int y = 0; y < Z2; y++) {
            weight2[x][y] = rand() / (double)(RAND_MAX);
        }
        b2[x] = rand() / (double)(RAND_MAX);
    }
    for (int x = 0; x < Z2; x++) {
        for (int y = 0; y < OUTPUT; y++) {
            weight3[x][y] = rand() / (double)(RAND_MAX);
        }
        b3[x] = rand() / (double)(RAND_MAX);
    }
    
    
    vector<tuple<vector<double>, vector<double>>> input_file;
    
    
    ifstream myFile ("/Users/mr.green/Desktop/xor_data.csv");
    if (myFile.is_open()) {
        double temp1;
        int temp2;
        while (myFile >> temp1) {
            
            vector<double> in;
            in.push_back(temp1);
            
            for(int x = 0; x < INPUT-1; x++) {
                myFile >> temp1;
                in.push_back(temp1);
            }
            
            vector<double> out(OUTPUT, 0);
            myFile >> temp2;
            out[temp2] = 1;
            
            
            input_file.push_back(make_tuple(in, out));
        }
        myFile.close();
    } else {
        cout << "Unable to Open file...\n";
    }
    
    
    int epoch = 0;
    while (epoch++ < EPOCH) {
        vector<double> z1(Z1, 0), z2(Z2, 0), z3(OUTPUT, 0);
        
        
        tuple<vector<double>, vector<double>> temp = input_file[epoch % input_file.size()];
        vector<double> input = get<0>(temp);
        vector<double> exp_out = get<1>(temp);
        
        //MARK: Forward
        for (int x = 0; x < INPUT; x++) {
            for (int y = 0; y < Z1; y++) {
                z1[y] += weight1[x][y] * input[x] + b1[y];
            }
        }
        vector<double> a1(Z1, 0);
        for (int x = 0; x < Z1; x++) {
            a1[x] = relu(z1[x]);
        }
        
        
        for (int x = 0; x < Z1; x++) {
            for (int y = 0; y < Z2; y++) {
                z2[y] += a1[x] * weight2[x][y] + b2[y];
            }
        }
        vector<double> a2(Z2, 0);
        for (int x = 0; x < Z2; x++) {
            a2[x] = relu(z2[x]);
        }
        
        
        for (int x = 0; x < Z2; x++) {
            for (int y = 0; y < OUTPUT; y++) {
                z3[y] += a2[x] * weight3[x][y] + b3[y];
            }
        }
        vector<double> a3(OUTPUT, 0);
        a3 = softmax(z3);
        
        
        //MARK: Backward
        vector<double> c_â3(OUTPUT, 0);
        for (int y = 0; y < OUTPUT; y++) {
            c_â3[y] = 2 * (a3[y] - exp_out[y]);
        }
        vector<double> â_z3 = softmax_derivative(z3);
        vector<double> z_w3 = a2;
        vector<vector<double>> c_w3(OUTPUT, vector<double>(Z2, 0));
        for (int x = 0; x < OUTPUT; x++) {
            for (int y = 0; y < Z2; y++) {
                c_w3[x][y] = c_â3[x] * â_z3[x] * z_w3[y];
            }
        }
        vector<double> c_b3(OUTPUT, 0);
        for (int x = 0; x < OUTPUT; x++) {
            c_b3[x] = c_â3[x] * â_z3[x];
        }
        
        
        vector<double> c_â2(Z2, 0);
        for (int x = 0; x < OUTPUT; x++) {
            for (int y = 0; y < Z2; y++) {
                c_â2[y] += weight3[y][x] * c_b3[x];
            }
        }
        vector<double> â_z2(Z2, 0);
        for (int y = 0; y < Z2; y++) {
            â_z2[y] = relu_derivative(z2[y]);
        }
        vector<double> z_w2 = a1;
        vector<vector<double>> c_w2(Z2, vector<double>(Z1, 0));
        for (int x = 0; x < Z2; x++) {
            for (int y = 0; y < Z1; y++) {
                c_w2[x][y] = c_â2[x] * â_z2[x] * z_w2[y];
            }
        }
        vector<double> c_b2(Z2, 0);
        for (int x = 0; x < Z2; x++) {
            c_b2[x] = c_â2[x] * â_z2[x];
        }
        
        
        vector<double> c_â1(Z1, 0);
        for (int y = 0; y < Z1; y++) {
            for (int x = 0; x < Z2; x++) {
                c_â1[y] += weight2[x][y] * c_b2[x];
            }
        }
        vector<double> â_z1(Z1, 0);
        for (int y = 0; y < Z1; y++) {
            â_z1[y] = relu_derivative(z1[y]);
        }
        vector<double> z_w1 = input;
        vector<vector<double>> c_w1(Z1, vector<double>(INPUT, 0));
        for (int x = 0; x < Z1; x++) {
            for (int y = 0; y < INPUT; y++) {
                c_w1[x][y] = c_â1[x] * â_z1[x] * z_w1[y];
            }
        }
        vector<double> c_b1(Z1, 0);
        for (int x = 0; x < Z1; x++) {
            c_b1[x] = c_â1[x] * â_z1[x];
        }
        
        
        //MARK: Change weights
        for (int x = 0; x < Z2; x++) {
            for (int y = 0; y < OUTPUT; y++) {
                weight3[x][y] -= c_w3[y][x] * LEARN_RATE;
            }
        }
        for (int x = 0; x < Z1; x++) {
            for (int y = 0; y < Z2; y++) {
                weight2[x][y] -= c_w2[y][x] * LEARN_RATE;
            }
        }
        for (int x = 0; x < INPUT; x++) {
            for (int y = 0; y < Z1; y++) {
                weight1[x][y] -= c_w1[y][x] * LEARN_RATE;
            }
        }
        
        //MARK: Change bias
        for (int x = 0; x < OUTPUT; x++) {
            b3[x] -= c_b3[x] * LEARN_RATE;
        }
        for (int x = 0; x < Z2; x++) {
            b2[x] -= c_b2[x] * LEARN_RATE;
        }
        for (int x = 0; x < Z1; x++) {
            b1[x] -= c_b1[x] * LEARN_RATE;
        }
        
        //MARK: ERROR
        double error = 0;
        for (int y = 0; y < OUTPUT; y++) {
            double temp = a3[y] - exp_out[y];
            error += temp * temp;
        }
        error /= OUTPUT;
        
        if (epoch % 50 == 0) {
            error_vec.push_back(error);
        }
        
        if (epoch % 1000 == 0) {
            cout << error << endl;
        }
    }
    
    
    //MARK: TEST DNN
    cout << endl << "Test your Neural Network with [x , y]" << endl;
    cout << "X = 1 or 0" << endl;
    cout << "Y = 1 or 0" << endl;
    for (int x = 0; x < 4; x++) {
        vector<double> z1(Z1, 0), z2(Z2, 0), z3(OUTPUT, 0);
        vector<double> temp_2;
        double num1;
        
        for(int x = 0; x < INPUT; x++) {
            cin >> num1;
            temp_2.push_back(num1);
        }
        
        
        for (int x = 0; x < INPUT; x++) {
            for (int y = 0; y < Z1; y++) {
                z1[y] += weight1[x][y] * temp_2[x] + b1[y];
            }
        }
        
        
        vector<double> a1(Z1);
        for (int x = 0; x < Z1; x++) {
            a1[x] = relu(z1[x]);
        }
        
        for (int x = 0; x < Z1; x++) {
            for (int y = 0; y < Z2; y++) {
                z2[y] += a1[x] * weight2[x][y] + b2[y];
            }
        }
        
        vector<double> a2(Z2);
        for (int x = 0; x < Z2; x++) {
            a2[x] = relu(z2[x]);
        }
        
        for (int x = 0; x < Z2; x++) {
            for (int y = 0; y < OUTPUT; y++) {
                z3[y] += a2[x] * weight3[x][y] + b3[y];
            }
        }
        
        vector<double> a3 = softmax(z3);
        
        cout << "Result: " << "[ " << a3[0];
        
        for(int x = 1; x < OUTPUT; x++) {
            cout << ", " << a3[x];
        }
        cout << " ]" << endl;
        
    }
    
    return 0;
}
