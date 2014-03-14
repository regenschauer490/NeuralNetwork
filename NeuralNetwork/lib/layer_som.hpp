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
		using NodeData_ = std::array<double, RefVecDim>;
		using DEdge_ = DirectedEdge<NodeData_>;
		using Node_ = Node<NodeData_, DEdge_>;
		using NodePtr_ = NodePtr<NodeData_, DEdge_>;
		using C_NodePtr_ = C_NodePtr<NodeData_, DEdge_>;
		using LayerPtr_ = SOMLayerPtr<RefVecDim>;

	private:
		std::vector<NodePtr_> nodes_;	//矩形を行で直列化. [row*COL + col]でアクセス

//		SIG_FRIEND_WITH_SOMLAYER

	public:
		const uint row_num_;
		const uint col_num_;

	private:
		LayerPtr_ CloneImpl() const{ return LayerPtr_(new SOMLayer<RefVecDim>(this->row_num_, this->col_num_)); }

		NodePtr_ Access(uint row, uint col){ return nodes_[row * col_num_ + col]; }

		auto begin() ->decltype(nodes_.begin()){ return nodes_.begin(); }
		auto end() ->decltype(nodes_.end()){ return nodes_.end(); }

	public:
		SOMLayer(uint row_num, uint col_num);

		//static LayerPtr_ MakeInstance(uint dim){ return std::shared_ptr<SOMLayer>(new SOMLayer(dim)); }

		LayerPtr_ CloneInitInstance() const{ return CloneImpl(); }

		uint NodeNum() const{ return row_num_ * col_num_; }

		C_NodePtr_ Access(uint row, uint col) const{ return nodes_[row * col_num_ + col]; }

		auto begin() const ->decltype(nodes_.cbegin()){ return nodes_.cbegin(); }
		auto end() const ->decltype(nodes_.cend()){ return nodes_.cend(); }
	};


	template <size_t RefVecDim>
	SOMLayer<RefVecDim>::SOMLayer(uint row_num, uint col_num)
		: row_num_(row_num), col_num_(col_num), nodes_(std::vector<NodePtr_>(row_num * col_num))
	{
		// ノード間距離計算
		auto CalcEdgeCost = [&](NodePtr_ const& nd, NodePtr_ const& na){
			using NVal = decltype(*nd->Score().begin());
			auto delta = sig::ZipWith([&](NVal vd, NVal va){ return sig::DeltaAbs(va, vd); }, nd->Score(), na->Score());
			return std::accumulate(std::begin(delta), std::end(delta), 0.0);
		};

		for (uint rd = 0; rd<row_num; ++rd){
			for (uint cd = 0; cd<col_num; ++cd){
				for (uint ra = rd; ra<row_num; ++ra){
					for (uint ca = cd; ca<col_num; ++ca){
						signn::Connect(
							nodes_[rd * col_num_ + cd],
							nodes_[ra * col_num_ + ca],
							std::make_shared<DEdge_>(CalcEdgeCost(nodes_[rd * col_num_ + cd], nodes_[ra * col_num_ + ca]))
						);
					}
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