#include<stdio.h>  
#include<string.h>  
#include<stdlib.h>  
#include<math.h>  
#include<vector>  
#include<iostream>  
/*
实现逻辑回归算法

参考自 http://blog.csdn.net/u014403897/article/details/45871939
*/
using namespace std;

// LR函数
double hypothesis(vector<double> & feature, vector<double>& w){
	double sum = 0.0;
	for (int i = 0; i < feature.size(); ++i)
		sum += w[i] * feature[i];
	// Sigmoid 函数
	return 1 / (1 + exp(-sum));
}

// 代价函数
double cost_function(vector<vector<double> > &feature_sample, vector<double> &w, vector<double>& label){
	double sum = 0.0;
	for (int i = 0; i < label.size(); ++i)
		sum += -label[i] * log(hypothesis(feature_sample[i], w)) - (1 - label[i]) * log(1 - hypothesis(feature_sample[i], w));

	return sum / label.size();
}

// LR算法实现过程
void logic_regression(vector<vector<double> > &feature_sample, vector<double>& label, vector<double> &w, double a){
	vector<double> delta_w;
	for (int j = 0; j < feature_sample[0].size(); ++j){
		double sum = 0.0;
		for (int i = 0; i < label.size(); ++i){
			sum += (hypothesis(feature_sample[i], w) - label[i]) * feature_sample[i][j];
		}
		delta_w.push_back(sum / label.size() * a);
	}
	for (int i = 0; i < w.size(); ++i)
		w[i] -= delta_w[i];
	cout << cost_function(feature_sample, w, label) << endl;
}

int main(){
	FILE *stream;
	freopen_s(&stream, "in.txt", "r", stdin);
	int feature_num, training_num, t;
	double a;
	cin >> feature_num >> training_num >> a >> t;
	vector<vector<double> >feature_sample;
	vector<double> tem;
	vector<double> lable;
	vector<double> w(feature_num+1, 0);
	double m;
	for (int i = 0; i<training_num; i++){
		tem.clear();
		tem.push_back(1);
		for (int j = 0; j<feature_num; j++){
			cin >> m;
			tem.push_back(m);
		}
		cin >> m;
		lable.push_back(m);
		feature_sample.push_back(tem);
	}
	/*for (int i = 0; i <= feature_num; i++) 
		w.push_back(0);*/
	while (t--) 
		logic_regression(feature_sample, lable, w, a);
	for (int i = 0; i <= feature_num; i++) 
		cout << w[i] << " ";

	system("pause");
	return 0;
}