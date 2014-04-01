/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_SOM_H
#define SIG_NN_LAYER_SOM_H

#include "edge.hpp"

namespace signn{

	template <size_t RefVecDim>
	class SOMLayer
	{
	public:
		using NodeData_ = sig::Array<double, RefVecDim>;
		using DEdge_ = DirectedEdge<NodeData_>;
		using Node_ = Node<NodeData_, DEdge_>;
		using NodePtr_ = NodePtr<NodeData_, DEdge_>;
		using C_NodePtr_ = C_NodePtr<NodeData_, DEdge_>;
		using LayerPtr_ = SOMLayerPtr<RefVecDim>;

	private:
		// 2次元の配置を行で直列化. [row*COL + col]でアクセス
		// node間を繋ぐedgeのweightは、通常の重みではなくステップ数を表す
		std::vector<NodePtr_> nodes_;	

		// SOMレイヤーの座標値に変換する際の修正値(正方形では補正なし、ハニカムでは偶数番目に+0.5)
		const double pos_col_offset;

	public:
		const uint row_num_;
		const uint col_num_;

	private:
		LayerPtr_ CloneImpl() const{ return LayerPtr_(new SOMLayer<RefVecDim>(this->row_num_, this->col_num_)); }

		NodePtr_ Access(uint row, uint col){ return nodes_[row * col_num_ + col]; }

		double Distance(std::array<uint,2> const& n1, std::array<uint,2> const& n2) const{ return sigdm::EuclideanDistance()(n1, n2); }	//矩形の場合

		auto begin() ->decltype(nodes_.begin()){ return nodes_.begin(); }
		auto end() ->decltype(nodes_.end()){ return nodes_.end(); }

	public:
		SOMLayer(uint row_num, uint col_num);

		// static LayerPtr_ MakeInstance(uint dim){ return std::shared_ptr<SOMLayer>(new SOMLayer(dim)); }

		LayerPtr_ CloneInitInstance() const{ return CloneImpl(); }

		uint NodeNum() const{ return row_num_ * col_num_; }

		C_NodePtr_ Access(uint row, uint col) const{ return nodes_[row * col_num_ + col]; }

		// 指定ノード間の2次元マップ上での距離を返す
		double Distance(C_NodePtr_ node1, C_NodePtr_ node2) const;

		// 指定ノードの2次元マップ上での座標を返す
		auto Position(C_NodePtr_ target) const->std::array<uint, 2>;

		auto begin() const ->decltype(nodes_.cbegin()){ return nodes_.cbegin(); }
		auto end() const ->decltype(nodes_.cend()){ return nodes_.cend(); }
	};


	template <size_t RefVecDim>
	SOMLayer<RefVecDim>::SOMLayer(uint row_num, uint col_num)
		: row_num_(row_num), col_num_(col_num), pos_col_offset(0.0)
	{
		sig::SimpleRandom<double> random(2.0, 8.0, DEBUG_MODE);

		for (uint i = 0; i < row_num * col_num; ++i){
			auto node = std::make_shared<Node_>();
			sig::Array<double, RefVecDim> init_score;
			for (uint j = 0; j<RefVecDim; ++j) init_score.push_back(random());
			node->Score(init_score);
			nodes_.push_back(node);
		}
		
		for (uint rd = 0; rd<row_num; ++rd){
			for (uint cd = 0; cd<col_num; ++cd){
				for (uint ra = 0; ra<row_num; ++ra){
					for (uint ca = 0; ca<col_num; ++ca){
						if (rd == ra && cd == ca) continue;

						signn::Connect(
							nodes_[rd * col_num_ + cd],
							nodes_[ra * col_num_ + ca],
							std::make_shared<DEdge_>(std::abs(rd - ra) + std::abs(cd - ca));
						);
					}
				}
			}
		}
	}

	template <size_t RefVecDim>
	double SOMLayer<RefVecDim>::Distance(C_NodePtr_ node1, C_NodePtr_ node2) const
	{
		using NVal = decltype(*nd->Score().begin());
		auto delta = sig::ZipWith([&](NVal vd, NVal va){ return sig::DeltaAbs(va, vd); }, );
		return std::accumulate(std::begin(delta), std::end(delta), 0.0);
	}

	template <size_t RefVecDim>
	auto SOMLayer<RefVecDim>::Position(C_NodePtr_ target) const->std::array<uint, 2>
	{
		for (uint r = 0; r < row_num_; ++r){
			for (uint c = 0; c < col_num_; ++c){
				if (target == nodes_[r * col_num_ + c]){
					return {{ r, r % 2 == 0 ? c + pos_col_offset : c }};	//座標値(y, x)
				}
			}
		}
	}

	/*
	class HoneycamLayer : public SOMLayer
	{
	const uint node_num_;
	std::vector<NodePtr> nodes_;

	SIG_FRIEND_WITH_LAYER;
	private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new HoneycamLayer()); }

	public:
	HoneycamLayer(uint row_num, uint col_num) : Layer(row_num * col_num){}
	};
	*/
}
#endif