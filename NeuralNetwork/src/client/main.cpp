#include "../lib/MLP_Online.hpp"
#include "../lib/MLP_Batch.hpp"
#include "../lib/AutoEncoder.hpp"

#include "utility.hpp"

#define IS_BATCH 0

//回帰
void Test1(){
	using namespace signn;
	const uint MAX_LOOP = 1000000;
	const uint DNUM = 200;
	const uint VNUM = 5;
/*
	std::vector<std::vector<double>> train_data{ { 0.40, 0.20 }, { 0.30, 0.40 }, { 0.80, 0.10 }, { 0.00, 0.00 }, { 0.10, 0.70 }, { 0.10, 0.20 }, { 0.50, 0.50 }, { 0.60, 0.20 }, { 0.20, 0.80 } };
	std::vector<double> train_ans{ 0.60, 0.70, 0.90, 0.00, 0.80, 0.30, 1.00, 0.80, 1.00 };
	std::vector<std::vector<double>> test_data{ { 0.40, 0.10 }, { 0.20, 0.70 }, { 0.10, 0.10 }, { 0.40, 0.40 } };
	std::vector<double> test_ans{ 0.50, 0.90, 0.20, 0.80 };
*/
	typedef InputInfo<double, VNUM> InInfo;
	typedef OutputInfo<OutputLayerType::Regression, 1> OutInfo;
#if IS_BATCH
	typedef Perceptron_Batch<InInfo, OutInfo> Perceptron;
#else
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;
#endif
	auto mid = Layer::MakeInstance(2);
	Perceptron nn({ mid });

	auto MakeData = [](uint data_num, uint elem_num){
		static SimpleRandom<double> rgen(0.0, 1.0, true);

		std::vector<std::vector<double>> d; d.reserve(data_num);
		std::vector<double> a;	a.reserve(elem_num);

		for (uint i = 0; i < data_num; ++i){
			std::vector<double> vec;
			for (uint j = 0; j < elem_num; ++j){
				vec.push_back(rgen());
			}
			d.push_back(std::move(vec));

			a.push_back( std::accumulate(++d[i].begin(), d[i].end(), 0.0) );
		}

		return std::make_tuple(std::move(d), std::move(a));
	};

	auto CheckMSE = [](Perceptron const& nn, std::vector< std::vector<double>> const& test_data, std::vector<double> const& test_ans){
		double tmse = 0;
		for (uint k = 0; k < test_data.size(); ++k){
			auto tresult = nn.Test(nn.MakeInputData(test_data[k].begin(), test_data[k].end()));
			tmse += tresult->SquareError(test_ans[k]);
		}
		tmse /= test_data.size();
		std::cout << "test_mse:" << tmse << std::endl;
		return tmse;
	};

	auto train = MakeData(DNUM, VNUM);
	auto train_data = std::move(std::get<0>(train));
	auto train_ans = std::move(std::get<1>(train));
	
	auto test = MakeData(DNUM / 10 + 10, VNUM);
	auto test_data = std::move(std::get<0>(test));
	auto test_ans = std::move(std::get<1>(test));

	std::vector<Perceptron::InputDataPtr> inputs;
	for (int i = 0; i < train_ans.size(); ++i){
		inputs.push_back(nn.MakeInputData(train_data[i].begin(), train_data[i].end(), train_ans[i]));
	}

	double p_mse = -1, mse = -1;
	std::tuple<uint, double> mse_min{0, 1000000};
	auto tmse_min = mse_min;
	sig::TimeWatch tw;

	for (uint loop = 0; loop < MAX_LOOP; ++loop){
		std::vector<double> moe;
#if !IS_BATCH
		for (uint i = 0; i < inputs.size(); ++i){
			moe.push_back(nn.Learn(inputs[i], true));
		}
#else
		moe.push_back(nn.Learn(inputs));
#endif
		p_mse = mse;
		mse = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();
		if (mse < std::get<1>(mse_min)){
			std::get<0>(mse_min) = loop;
			std::get<1>(mse_min) = mse;
		}

		if (loop%1 == 0){
			auto tmse = CheckMSE(nn, test_data, test_ans);
			if (tmse < std::get<1>(tmse_min)){
				std::get<0>(tmse_min) = loop;
				std::get<1>(tmse_min) = tmse;
			}
			if (tmse < 0.0001) break;
		}

		if (std::abs(p_mse - mse)<0.00000001) break;
		//if (mse < 0.00005) break;
	}

	tw.Stop();
	std::cout << "time: " << tw.GetTime<std::chrono::seconds>() << std::endl;

	CheckMSE(nn, test_data, test_ans);
	
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

//多値分類(手書き文字識別)
void Test3(){
	using namespace signn;
	
	typedef bool input_type;
	typedef InputInfo<input_type, 784> InInfo;
	typedef OutputInfo<OutputLayerType::MultiClassClassification, 10> OutInfo;
#if IS_BATCH
	typedef Perceptron_Batch<InInfo, OutInfo> Perceptron;
#else
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;
#endif

	auto mid = Layer::MakeInstance(100);

	Perceptron nn(std::vector<LayerPtr>{mid});

	//nn.LoadParameter(L"test data/");

	std::vector<std::vector<input_type>> train_data;
	std::vector<int> train_ans;

	for (int doc = 0; doc <10; ++doc){
		auto rows = *sig::File::ReadLine<std::string>(L"test data/train" + std::to_wstring(doc) + L".txt");
		
		for (uint r=0; r<rows.size(); ++r){
			train_data.push_back(std::vector<input_type>());
			auto split = sig::String::Split(rows[r], ",");
			train_ans.push_back(std::stoi(split[0]));
			std::transform(++split.begin(), split.end(), std::back_inserter(train_data.back()), [](std::string s){ return std::stoi(s); });
		}
	}

	sig::Shuffle(train_data, train_ans);
	
	std::vector<std::vector<input_type>> test_data;
	std::vector<int> test_ans;
	uint tds;
	for (tds = train_data.size() - 1; tds > train_data.size() - 15; --tds){
		test_data.push_back(train_data[tds]);
		test_ans.push_back(train_ans[tds]);
	}
	train_data.resize(tds + 1);
	train_ans.resize(tds + 1);

#if IS_BATCH
	const uint DATA_DIV = 10;
	const uint DATA_DIV_DELTA = train_data.size() / DATA_DIV;
	std::vector<std::vector<Perceptron::InputDataPtr>> inputs(DATA_DIV);
	for (uint div = 0; div < DATA_DIV; ++div){
		for (uint i = DATA_DIV_DELTA*div; i < DATA_DIV_DELTA*(div+1); ++i){
			inputs[div].push_back(nn.MakeInputData(train_data[i].begin(), train_data[i].end(), train_ans[i]));
		}
	}
#else
	std::vector<Perceptron::InputDataPtr> inputs;
	for (uint i = 0; i < train_ans.size(); ++i){
		inputs.push_back(nn.MakeInputData(train_data[i].begin(), train_data[i].end(), train_ans[i]));
	}
#endif

	std::vector<Perceptron::InputDataPtr> test_inputs;
	for (auto const& td : test_data) test_inputs.push_back(nn.MakeInputData(td.begin(), td.end()));

	double p_esum = 0, esum = 0;
	for (int loop = 0; true; ++loop){
		sig::TimeWatch tw;
		std::vector<double> moe;
#if !IS_BATCH
		for (int i = 0; i < inputs.size(); ++i){
			moe.push_back(nn.Learn(inputs[i], true));
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();
#else
		for (uint div = 0; div < DATA_DIV; ++div){
			p_esum = esum;
			esum = nn.Learn(inputs[div]);
#endif
		//nn.SaveParameter(L"test data/");

		tw.Stop();
		std::cout << "time: " << tw.GetTime<std::chrono::seconds>() << std::endl;
		if (loop % 1 == 0) std::cout << "train_mse: " << esum << std::endl << std::endl;

		for (int i=0; i<inputs.size(); ++i){
			auto est = nn.Test(test_inputs[i]);
			for (uint j = 0; j < est->size(); ++j){
				if ((*est)[j]) std::cout << j << ", ";
			}
			std::cout << " ans:" << test_ans[i] << std::endl;
		}
#if IS_BATCH
			tw.ReStart();
		}
#endif
		//if (esum < 1000) break;
		if (std::abs(p_esum - esum) < 0.0000000001) break;
	}
}

//オートエンコーダ
void Test4()
{
	using namespace signn;

	typedef bool input_type;
	typedef InputInfo<input_type, 784> InInfo;
	typedef AutoEncoder<InInfo, 10> AutoEncoder;

	AutoEncoder ae;

	std::vector < std::vector<input_type>> train_data;

	auto rows = *sig::File::ReadLine<std::string>(L"test data/train0.txt");

	for (uint r = 0; r < rows.size(); ++r){
		train_data.push_back(std::vector<input_type>());
		auto split = sig::String::Split(rows[r], ",");
		std::transform(++split.begin(), split.end(), std::back_inserter(train_data.back()), [](std::string s){ return std::stoi(s); });
	}
	
	sig::Shuffle(train_data);

	std::vector<std::vector<input_type>> test_data;
	uint tds;
	for (tds = train_data.size() - 1; tds > train_data.size() - 15; --tds){
		test_data.push_back(train_data[tds]);
	}
	train_data.resize(tds + 1);
	
	std::vector<AutoEncoder::InputDataPtr> inputs;
	for (auto const& td : train_data){
		inputs.push_back(ae.MakeInputData(td.begin(), td.end()));
	}

	std::vector<AutoEncoder::InputDataPtr> test_inputs;
	for (auto const& td : test_data) test_inputs.push_back(ae.MakeInputData(td.begin(), td.end()));


	double p_esum = 0, esum = 0;
	for (int loop = 0; true; ++loop){
		sig::TimeWatch tw;
		std::vector<double> moe;

		for (int i = 0; i < inputs.size(); ++i){
			moe.push_back(ae.Learn(inputs[i], true));
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();
		//nn.SaveParameter(L"test data/");

		tw.Stop();
		std::cout << "time: " << tw.GetTime<std::chrono::seconds>() << std::endl;
		if (loop % 1 == 0) std::cout << "train_mse: " << esum << std::endl << std::endl;

		for (int i = 0; i < inputs.size(); ++i){
			auto est = *ae.Test(test_inputs[i]);
			sig::Histgram<double, 10> hist(-1, 1);

			for (uint j = 0; j < est.size(); ++j){
				hist.Count( CrossCorrelation(test_data[i].begin(), test_data[i].end(), est.begin(), est.end(), 0) );
			}
			hist.Print();
		}

		//if (esum < 1000) break;
		if (std::abs(p_esum - esum) < 0.0000000001) break;
	}
}


int main(){
	Test4();
}

