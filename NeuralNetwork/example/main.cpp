#include "../lib/MLP_online.hpp"
#include "../lib/MLP_batch.hpp"
#include "../lib/auto_encoder.hpp"
#include "../lib/SOM_online.hpp"

#include "../lib/external/SigUtil/lib/tool.hpp"
#include "../lib/external/SigUtil/lib/file.hpp"
#include "../lib/external/SigUtil/lib/modify.hpp"
#include "../lib/external/SigUtil/lib/string.hpp"

//#include "utility.hpp"

const std::wstring test_folder = L"test data/";

#define IS_BATCH 1

//回帰
void Test1(){
	using namespace signn;
	const uint MAX_LOOP = 1000000;
	const uint DNUM = 200;
	const uint VNUM = 2;

	typedef InputInfo<double, VNUM> InInfo;
	typedef OutputInfo<RegressionLayerInfo> OutInfo;
#if IS_BATCH
	typedef Perceptron_Batch<InInfo, OutInfo> Perceptron;
#else
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;
#endif
	auto mid = Perceptron::MakeMidLayer(2);
	Perceptron nn(learning_rate_sample, L2__regularization_sample, { mid });
	
	auto MakeData = [&](uint data_num, uint elem_num){
		static sig::SimpleRandom<double> rgen(0.0, 1.0, true);

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

	auto CheckMSE = [](Perceptron const& nn, std::vector< std::vector<double>> const& test_data, std::vector<double> const& test_ans, bool disp){
		double tmse = 0;
		for (uint k = 0; k < test_data.size(); ++k){
			auto tresult = nn.Test(nn.MakeInputData(test_data[k].begin(), test_data[k].end()));
			tmse += tresult->MeanSquareError(test_ans[k]);

			if(disp){
				std::cout << "est:" << (*tresult)[0] << ", ans:" << test_ans[k] << std::endl;
			}
		}
		tmse /= test_data.size();
		return tmse;
	};

	
	std::vector<std::vector<double>> train_data{ { 0.40, 0.20 }, { 0.30, 0.40 }, { 0.80, 0.10 }, { 0.00, 0.00 }, { 0.10, 0.70 }, { 0.10, 0.20 }, { 0.50, 0.50 }, { 0.60, 0.20 }, { 0.20, 0.80 } };
	std::vector<double> train_ans{ 0.60, 0.70, 0.90, 0.00, 0.80, 0.30, 1.00, 0.80, 1.00 };
	std::vector<std::vector<double>> test_data{ { 0.40, 0.10 }, { 0.20, 0.70 }, { 0.10, 0.10 }, { 0.40, 0.40 } };
	std::vector<double> test_ans{ 0.50, 0.90, 0.20, 0.80 };
	
/*
	auto train = MakeData(DNUM, VNUM);
	auto train_data = std::move(std::get<0>(train));
	auto train_ans = std::move(std::get<1>(train));
	
	auto test = MakeData(DNUM / 10 + 10, VNUM);
	auto test_data = std::move(std::get<0>(test));
	auto test_ans = std::move(std::get<1>(test));
*/
	std::vector<Perceptron::InputDataPtr> inputs;
	for (uint i = 0; i < train_ans.size(); ++i){
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
			moe.push_back(nn.Train(inputs[i], true));
		}
#else
		moe.push_back(nn.Train(inputs));
#endif
		p_mse = mse;
		mse = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();
		if (mse < std::get<1>(mse_min)){
			std::get<0>(mse_min) = loop;
			std::get<1>(mse_min) = mse;
		}

		if (loop%100 == 0){
			auto tmse = CheckMSE(nn, test_data, test_ans, false);
			if (tmse < std::get<1>(tmse_min)){
				std::get<0>(tmse_min) = loop;
				std::get<1>(tmse_min) = tmse;
			}
			if (tmse < 0.001) break;

			std::cout << "test_mse:" << tmse << "	,mse:" << mse << std::endl;
		}

		//if (std::abs(p_mse - mse)<0.00000001) break;
		//if (mse < 0.00005) break;
	}

	tw.Save();
	std::cout << "time: " << tw.GetTotalTime<std::chrono::seconds>() << std::endl;

	CheckMSE(nn, test_data, test_ans, true);

	nn.SaveParameter(L"test data/dst", false);
	nn.SaveParameter(L"test data/opt", true);
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
			moe.push_back(nn.Train(Perceptron::InputData(train_data[i].begin(), train_data[i].end(), train_ans[i])));
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
	const uint MAX_LOOP = 1000000;
	
	typedef bool input_type;
	typedef InputInfo<input_type, 784> InInfo;
	typedef OutputInfo<MultiClassClassifyLayerInfo<10>> OutInfo;
#if IS_BATCH
	typedef Perceptron_Batch<InInfo, OutInfo> Perceptron;
#else
	typedef Perceptron_Online<InInfo, OutInfo> Perceptron;
#endif

	auto mid = Perceptron::MakeMidLayer(100);

	Perceptron nn(learning_rate_sample, L2__regularization_sample, {mid});

	//nn.LoadParameter(L"test data/opt");

	std::vector<std::vector<input_type>> train_data;
	std::vector<int> train_ans;

	for (int doc = 0; doc <10; ++doc){
		auto rows = *sig::ReadLine<std::string>(test_folder + L"mlp/train" + std::to_wstring(doc) + L".txt");
		
		for (uint r=0; r<rows.size(); ++r){
			train_data.push_back(std::vector<input_type>());
			auto split = sig::Split(rows[r], ",");
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

	long long total_time = 0;
	double p_esum = 0, esum = 0;
	for (int loop = 0; loop < MAX_LOOP; ++loop){
		sig::TimeWatch tw;
		std::vector<double> moe;
#if !IS_BATCH
		for (uint i = 0; i < inputs.size(); ++i){
			moe.push_back(nn.Train(inputs[i], true));
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();
#else
		for (uint div = 0; div < DATA_DIV; ++div){
			p_esum = esum;
			esum = nn.Train(inputs[div]);
#endif
			
		tw.Save();
		total_time += tw.GetTotalTime<std::chrono::seconds>();
		if (loop%1 == 0){
			std::cout << "loop: " << loop << " ,time: " << tw.GetTotalTime<std::chrono::seconds>() << " ,total: " << total_time << std::endl << std::endl;

			double test_esum = 0;
			for (uint i=0; i< test_inputs.size(); ++i){
				auto est = nn.Test(test_inputs[i]);
				for (uint j = 0; j < est->size(); ++j){
					if ((*est)[j]) std::cout << j << ", ";
				}
				test_esum += est->MeanSquareError(test_ans[i]);
				std::cout << " ans:" << test_ans[i] << std::endl;
			}
			std::cout << "train_mse: " << esum << std::endl;
			std::cout << "test_mse: " << test_esum/test_inputs.size() << std::endl << std::endl;
		}
#if IS_BATCH
			tw.ReStart();
		}
#endif
		//if (esum < 1000) break;
		nn.SaveParameter(test_folder + L"mlp/dst", false);
		nn.SaveParameter(test_folder + L"mlp/opt", true);
		if (std::abs(p_esum - esum) < 0.0000000001) break;
	}
}

//オートエンコーダ（手書き文字復元）
void Test4()
{
	using namespace signn;

	typedef bool input_type;
	typedef InputInfo<input_type, 784> InInfo;
	typedef AutoEncoder<InInfo, 5> AutoEncoder;

	AutoEncoder ae(learning_rate_sample, L2__regularization_sample);

	//テストデータ読み込み
	std::vector < std::vector<input_type>> train_data;

	auto rows = *sig::ReadLine<std::string>(test_folder + L"mlp/train2.txt");
	for (uint r = 0; r < rows.size(); ++r){
		train_data.push_back(std::vector<input_type>());
		auto split = sig::Split(rows[r], ",");
		std::transform(++split.begin(), split.end(), std::back_inserter(train_data.back()), [](std::string s){ return std::stoi(s); });
	}
	
	//sig::Shuffle(train_data);

	std::vector<std::vector<input_type>> test_data;
	uint tds;
	for (tds = train_data.size() - 1; tds > train_data.size() - 15; --tds){
		test_data.push_back(train_data[tds]);
	}
	train_data.resize(tds + 1);
	
	//入力用データ作成
	std::vector<AutoEncoder::InputDataPtr> inputs;
	for (auto const& td : train_data){
		inputs.push_back(ae.MakeInputData(td.begin(), td.end()));
	}

	std::vector<AutoEncoder::InputDataPtr> test_inputs;
	for (auto const& td : test_data) test_inputs.push_back(ae.MakeInputData(td.begin(), td.end()));

	//学習開始
	double p_esum = 0, esum = 0;
	for (int loop = 0; true; ++loop){
		sig::TimeWatch tw;
		std::vector<double> moe;

		for (int i = 0; i < inputs.size(); ++i){
			moe.push_back(ae.Train(inputs[i], true));
			if (i>0 && std::abs(moe[i-1] - moe[i]) < 0.0000000001){
				std::cout << "iteration:" << i+1 << " / " << inputs.size() << std::endl;
				break;
			}
		}
		p_esum = esum;
		esum = std::accumulate(moe.begin(), moe.end(), 0.0) / train_data.size();

		tw.Stop();
		std::cout << "\n\ntime: " << tw.GetTotalTime<std::chrono::seconds>() << std::endl;
		std::cout << "train_mse: " << esum << std::endl << std::endl;

		ae.SaveParameter(test_folder + L"mlp/dst", false);
		ae.SaveParameter(test_folder + L"mlp/opt", true);

		if (loop % 1 == 0){
			for (int i = 0; i < test_inputs.size(); ++i){
				auto est = *ae.Test(test_inputs[i]);
				sig::Histgram<double, 10> hist(0, 0.5);
				std::vector<int> r;

				for (uint j = 0; j < est.size(); ++j){
					hist.Count( signn::Simirarlity(test_data[i].begin(), test_data[i].end(), est.begin(), est.end(), j) );
					r.push_back(est[j]);
				}
				hist.Print();
				sig::SaveNum(test_data[i], L"test data/test0_ans.txt", "", sig::WriteMode::overwrite);
				sig::SaveNum(r, L"test data/test0.txt", "", sig::WriteMode::overwrite);
			}
		}

		//if (esum < 1000) break;
		if (std::abs(p_esum - esum) < 0.0000000001) break;
	}
}

//自己組織化マップ
void Test5()
{
	using namespace signn;
	const uint ITERATION = 1;				//データ全体を学習する反復を何回するか

	const uint SOM_NODE_SQUARE = 10;		//SOMレイヤーの1辺のノード数
	using InInfo = InputInfo<double, 4>;
	using SOM = signn::SOM_Online<InInfo, SOM_NODE_SQUARE>;

	SOM som;

	//テストデータ読み込み
	auto test_raw_data = sig::ReadNum<std::vector<std::vector<double>>>(test_folder + L"som/test_iris.csv", ",");
	if (!test_raw_data){
		std::cout << "test-folder pass error" << std::endl;
		assert(false);
	}

	std::vector<int> train_ans;

	for (auto& row : *test_raw_data){
		train_ans.push_back(static_cast<int>(row[4]));
		row.pop_back();
	}
	auto train_data = std::move(*test_raw_data);
	const auto data_size = train_data.size();

	//入力用データ作成
	std::vector<SOM::InputDataPtr> inputs;
	
	for (auto i=0; i<data_size; ++i){
		inputs.push_back( som.MakeInputData(train_data[i].begin(), train_data[i].end(), train_ans[i]) );
	}

	//学習開始
	for (int loop = 0; loop < ITERATION; ++loop){
		for(auto const& input : inputs) som.Train(input);
	}

	auto pos = som.NearestPosition(inputs[0]);
}


int main(){
	Test5();
}

