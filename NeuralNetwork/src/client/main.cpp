
#include "../lib/MLP_Online.hpp"
#include "../lib/MLP_Batch.hpp"

#include "utility.hpp"


void Test1(){
	using namespace signn;

	std::vector<std::vector<double>> zero{ { 0.40, 0.20 }, { 0.30, 0.40 }, { 0.80, 0.10 }, { 0.00, 0.00 }, { 0.10, 0.70 }, { 0.10, 0.20 }, { 0.50, 0.50 }, { 0.60, 0.20 }, { 0.20, 0.80 } };
	std::vector<std::vector<double>> test{{0.50, 0.10}, {0.20, 0.70}, {0.10, 0.10}};

	std::vector<double> ans{ 0.60, 0.70, 0.90, 0.00, 0.80, 0.30, 1.00, 0.80, 1.00 };

	typedef InputInfo<double, 2> InInfo;
	typedef OutputInfo<OutputLayerType::Regression, 1> OutInfo;
	typedef Perceptron_Batch<InInfo, OutInfo> Perceptron;

	auto mid = Layer::MakeInstance(2);

	std::vector<Perceptron::InputData> inputs;
	for (int i = 0; i < ans.size(); ++i){
		inputs.push_back(Perceptron::InputData(zero[i].begin(), zero[i].end(), ans[i]));
	}

	sig::TimeWatch tw;

	Perceptron nn(std::vector<LayerPtr>{mid});

//	std::vector<double> weight{ -1.10566906, 1.10261568, -1.16590520, 1.18427171, -13.39621577, 7.56436575 };
//	nn.DebugWeight(weight);
//	auto result = nn.Test(test[0].begin(), test[0].end())->GetScore();
	
	double p_esum = 0, esum = 0;
	for(int loop = 0; true; ++loop){
		std::vector<double> moe;
		/*
		for(int i=0; i<ans.size(); ++i){
			moe.push_back(nn.Learn(Perceptron::InputData(zero[i].begin(), zero[i].end(), ans[i])));
		}*/
		//inputs.push_back(Perceptron::InputData(zero[loop % 9].begin(), zero[loop % 9].end(), ans[loop % 9]));
		moe.push_back(nn.Learn(inputs));

		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0);
		if (loop%100 == 0) std::cout << esum  << std::endl;
		if (std::abs(p_esum-esum)<0.0000000001) break;
		if (esum < 0.00005) break;
	}

	tw.Stop();
	std::cout << "time: " << tw.GetTime<std::chrono::seconds>() << std::endl;

	auto result1 = nn.Test(test[0].begin(), test[0].end())->GetScore();
	auto result2 = nn.Test(test[1].begin(), test[1].end())->GetScore();
	auto result3 = nn.Test(test[2].begin(), test[2].end())->GetScore();

	std::cout << result1[0] << ", " << result2[0] << ", " << result3[0];
}

/*
void Test2(){
	using namespace signn;

	typedef InputInfo<double, 784> InInfo;
	typedef OutputInfo<OutputLayerType::BinaryClassification, 1> OutInfo;
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;


	auto mid = Layer::MakeInstance(100);

	Perceptron nn(std::vector<LayerPtr>{mid});

	std::vector<std::vector<double>> train_data;
	std::vector<bool> train_ans;

	for (int doc = 0; doc < 10; ++doc){
		auto rows = *sig::File::ReadLine<std::string>(L"test data/train" + std::to_wstring(doc) + L".txt");
		for (auto const& row : rows){
			train_data.push_back(std::vector<double>());
			auto split = sig::Split(row, ",");
			train_ans.push_back(std::stoi(split[0]) == 0);
			std::transform(++split.begin(), split.end(), std::back_inserter(train_data.back()), [](std::string s){ return std::stod(s); });
		}
	}

	std::vector<std::vector<double>> test_data;
	std::vector<bool> test_ans;
	auto rows = *sig::File::ReadLine<std::string>(L"test data/test.txt");
	for (auto const& row : rows){
		test_data.push_back(std::vector<double>());
		auto split = sig::Split(row, ",");
		test_ans.push_back(std::stoi(split[0]) == 0);
		std::transform(++split.begin(), split.end(), std::back_inserter(test_data.back()), [](std::string s){ return std::stod(s); });
	}


	double p_esum = 0, esum = 0;
	for (int loop = 0; true; ++loop){
		std::vector<double> moe;
		for (int i = 0; i < train_data.size(); ++i){
			moe.push_back(nn.Learn(Perceptron::InputData(train_data[i].begin(), train_data[i].end(), train_ans[i])));
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0);
		std::cout << esum << std::endl;
		if (loop == 2) break;
		//if (sig::Equal(p_esum, esum)) break;
	}


	for (int i = 0; i < test_data.size(); ++i){
		auto est = nn.Test(test_data[i].begin(), test_data[i].end())->GetScore();
		std::cout << est[0] << ", " << test_ans[i] << std::endl;
	}
}
*/


void Test3(){
	using namespace signn;
		
	typedef InputInfo<int, 784> InInfo;
	typedef OutputInfo<OutputLayerType::MultiClassClassification, 10> OutInfo;
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;

	auto mid = Layer::MakeInstance(100);

	Perceptron nn(std::vector<LayerPtr>{mid});

	//nn.LoadParameter(L"test data/");

	std::vector<std::vector<int>> train_data;
	std::vector<int> train_ans;

	for (int doc = 0; doc <10; ++doc){
		auto rows = *sig::File::ReadLine<std::string>(L"test data/train" + std::to_wstring(doc) + L".txt");
		
		for (auto const& row : rows){
			train_data.push_back(std::vector<int>());
			auto split = sig::String::Split(row, ",");
			train_ans.push_back(std::stoi(split[0]));
			std::transform(++split.begin(), split.end(), std::back_inserter(train_data.back()), [](std::string s){ return std::stoi(s); });
		}
	}

	sig::Shuffle(train_data, train_ans);
	
	std::vector<std::vector<int>> test_data;
	std::vector<int> test_ans;
	uint tds;
	for (tds = train_data.size() - 1; tds > train_data.size() - 15; --tds){
		test_data.push_back(train_data[tds]);
		test_ans.push_back(train_ans[tds]);
	}
	train_data.resize(tds + 1);
	train_ans.resize(tds + 1);


	double p_esum = 0, esum = 0;
	for (int loop = 0; true; ++loop){
		std::vector<double> moe;
		for (int i = 0; i < train_data.size(); ++i){
			moe.push_back(nn.Learn(Perceptron::InputData(train_data[i].begin(), train_data[i].end(), train_ans[i])));
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0);
		nn.SaveParameter(L"test data/");

		for (int i=0; i<test_data.size(); ++i){
			auto est = nn.Test(test_data[i].begin(), test_data[i].end())->GetScore();
			for (uint j = 0; j < est.size(); ++j){
				if (est[j]) std::cout << j << ", ";
			}
			std::cout << " ans:" << test_ans[i] << std::endl;
		}

		if (loop % 1 == 0) std::cout << esum << std::endl;
		if (std::abs(p_esum - esum) < 0.0000000001) break;
		if (esum < 1000) break;
	}
}

int main(){
	Test3();
}

