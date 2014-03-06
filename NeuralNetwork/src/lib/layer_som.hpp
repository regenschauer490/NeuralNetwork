/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_SOM_H
#define SIG_NN_LAYER_SOM_H

#include "edge.h"

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
		const uint row_num_;
		const uint col_num_;
		std::vector<std::vector<NodePtr_>> nodes_;

		SIG_FRIEND_WITH_SOMLAYER

	private:
		SOMLayerPtr CloneImpl() const{ return std::make_shared<SOMLayer<RefVecDim>>(this->row_num_, this->col_num_)); }

		auto begin() ->decltype(nodes_.begin()){ return nodes_.begin(); }
		auto end() ->decltype(nodes_.end()){ return nodes_.end(); }

		NodePtr_ operator[](uint index){ return nodes_[index]; }

	public:
		SOMLayer(uint row_num, uint col_num);

		//static LayerPtr_ MakeInstance(uint node_num){ return std::shared_ptr<SOMLayer>(new SOMLayer(node_num)); }

		LayerPtr_ CloneInitInstance() const{ return CloneImpl(); }

		uint RowNum() const{ return row_num_; }
		uint ColNum() const{ return col_num_; }
		uint NodeNum() const{ return row_num_ * col_num_; }

		auto begin() const ->decltype(nodes_.cbegin()){ return nodes_.cbegin(); }
		auto end() const ->decltype(nodes_.cend()){ return nodes_.cend(); }

		C_NodePtr_ operator[](uint index) const{ return nodes_[index]; }
	};


	template <size_t RefVecDim>
	SOMLayer<RefVecDim>::SOMLayer(uint row_num, uint col_num)
		: row_num_(row_num), col_num_(col_num), nodes_(std::vector<std::vector<NodePtr>>(row_num, std::vector<NodePtr>(col_num)))
	{
		// ƒm[ƒhŠÔ‹——£ŒvŽZ
		auto CalcEdgeCost = [&](NodePtr_ const& nd, NodePtr_ const& na){
			using NVal = decltype(*nd->Score().begin());
			auto delta = sig::ZipWith([&](NVal vd, NVal va){ return sig::DeltaAbs(va, vd); }, nd->Score(), na->Score());
			return std::accumulate(std::begin(delta), std::end(delta), 0.0);
		};

		for (uint rd = 0; rd<row_num; ++rd){
			for (uint cd = 0; cd<col_num; ++cd){
				for (uint ra = rd; ra<row_num; ++ra){
					for (uint ca = cd; ca<col_num; ++ca){
						signn::Connect(nodes_[rd][cd], nodes_[ra][ca], std::make_shared<DEdge_>(CalcEdgeCost(nodes_[rd][cd], nodes_[ra][ca])));
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