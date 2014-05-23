/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_SOM_H
#define SIG_NN_LAYER_SOM_H

#include "node.hpp"

namespace signn{

	template <size_t RefVecDim>
	class SOMLayer
	{
	public:
		using ValueType_ = double;
		using NodeData_ = sig::array<ValueType_, RefVecDim>;
		using DEdge_ = DirectedEdge<NodeData_>;
		using Node_ = Node<NodeData_, DEdge_>;
		using NodePtr_ = NodePtr<NodeData_, DEdge_>;
		using C_NodePtr_ = C_NodePtr<NodeData_, DEdge_>;
		using LayerPtr_ = SOMLayerPtr<RefVecDim>;
		using DataRange_ = sig::array<std::pair<ValueType_, ValueType_>, RefVecDim>;

	private:
		// 2�����̔z�u���s�Œ���. [row*COL + col]�ŃA�N�Z�X
		// node�Ԃ��q��edge��weight�́A�ʏ�̏d�݂ł͂Ȃ��X�e�b�v����\��
		std::vector<NodePtr_> nodes_;	

		// SOM���C���[�̍��W�l�ɕϊ�����ۂ̏C���l(�����`�ł͕␳�Ȃ��A�n�j�J���ł͋����Ԗڂ�+0.5)
		const double pos_col_offset;

	public:
		const uint row_num_;
		const uint col_num_;

	private:
		LayerPtr_ CloneImpl() const{ return LayerPtr_(new SOMLayer<RefVecDim>(this->row_num_, this->col_num_)); }

		NodePtr_ Access(uint row, uint col){ return nodes_[row * col_num_ + col]; }

		double Distance(std::array<uint,2> const& n1, std::array<uint,2> const& n2) const{ return sigdm::EuclideanDistance()(n1, n2); }	//��`�̏ꍇ

		auto begin() ->decltype(nodes_.begin()){ return nodes_.begin(); }
		auto end() ->decltype(nodes_.end()){ return nodes_.end(); }

	public:
		SOMLayer(uint row_num, uint col_num, DataRange_ ref_vector_init);

		// static LayerPtr_ MakeInstance(uint dim){ return std::shared_ptr<SOMLayer>(new SOMLayer(dim)); }

		LayerPtr_ CloneInitInstance() const{ return CloneImpl(); }

		uint NodeNum() const{ return row_num_ * col_num_; }

		C_NodePtr_ Access(uint row, uint col) const{ return nodes_[row * col_num_ + col]; }

		// �w��m�[�h�Ԃ̃f�[�^��ԏ�(�Q�ƃx�N�g��)�̋���
		double VectorDistance() const{ return DFSelector<DistFUnc>::fobj(node1->Score(), node2->Score()); }

		// �w��m�[�h�Ԃ�2�����}�b�v��̋�����Ԃ�
		double Distance(C_NodePtr_ node1, C_NodePtr_ node2) const;

		// �w��m�[�h��2�����}�b�v��̍��W��Ԃ�
		auto Position(C_NodePtr_ target) const->std::array<uint, 2>;

		auto begin() const ->decltype(nodes_.cbegin()){ return nodes_.cbegin(); }
		auto end() const ->decltype(nodes_.cend()){ return nodes_.cend(); }
	};


	template <size_t RefVecDim>
	SOMLayer<RefVecDim>::SOMLayer(uint row_num, uint col_num, DataRange_ ref_vector_init)
		: row_num_(row_num), col_num_(col_num), pos_col_offset(0.0)
	{
		// �Q�ƃx�N�g���̏����l���e�����͈͓̔��Ń����_���ɐ���
		std::vector<sig::SimpleRandom<double>> random;
		for(auto init_range : ref_vector_init){
			auto min = std::get<0>(init_range) < std::get<1>(init_range) ? std::get<0>(init_range) : std::get<1>(init_range);
			auto max = std::get<0>(init_range) < std::get<1>(init_range) ? std::get<1>(init_range) : std::get<0>(init_range);
			random.push_back(sig::SimpleRandom<double>(min, max, DEBUG_MODE));
		}		
	
		for (uint i = 0; i < row_num * col_num; ++i){
			auto node = std::make_shared<Node_>();
			sig::array<double, RefVecDim> init_score;
			for (uint j = 0; j<RefVecDim; ++j) init_score.push_back(random[j]());

		/*	if (NeighborFunc == DistanceFunc::KL_Div || NeighborFunc == DistanceFunc::JS_Div){
				auto sum = std::accumulate(init_score.begin(), init_score.end(), 0.0);
				for (auto& e : init_score){
					e /= sum;		//���v���P�ɂȂ�悤���K��
				}
			}
		*/
			node->Score(init_score);
			nodes_.push_back(node);
		}
	
		//2�����}�b�v��m�[�h�̃����N����
		for (uint rd = 0; rd<row_num; ++rd){
			for (uint cd = 0; cd<col_num; ++cd){
				for (uint ra = 0; ra<row_num; ++ra){
					for (uint ca = 0; ca<col_num; ++ca){
						if (rd == ra && cd == ca) continue;

						signn::Connect(
							nodes_[rd * col_num_ + cd],
							nodes_[ra * col_num_ + ca],
							std::make_shared<DEdge_>(std::pow(std::labs(rd - ra) + std::labs(cd - ca), 2))
						);
					}
				}
			}
		}
	}

	template <size_t RefVecDim>
	double SOMLayer<RefVecDim>::Distance(C_NodePtr_ node1, C_NodePtr_ node2) const
	{
		auto pos1 = Position(node1);
		auto pos2 = Position(node2);

		std::inner_product(pos1.begin(), pos1.end(), pos2.begin(), 0.0, std::plus<ValueType_>(), );
		sig::ZipWith([](NodeData_ const& v1, NodeData_ const& v2){ }, pos1, pos2);
	}

	template <size_t RefVecDim>
	auto SOMLayer<RefVecDim>::Position(C_NodePtr_ target) const->std::array<uint, 2>
	{
		for (uint r = 0; r < row_num_; ++r){
			for (uint c = 0; c < col_num_; ++c){
				if (target == nodes_[r * col_num_ + c]){
					return {{ r, r % 2 == 0 ? c + pos_col_offset : c }};	//���W�l(y, x)
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