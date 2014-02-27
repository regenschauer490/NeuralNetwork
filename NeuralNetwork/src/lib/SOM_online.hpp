/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_ONLINE_H
#define SIG_NN_SOM_ONLINE_H

namespace signn{

class SOMLayer
{
	const uint row_num_;
	const uint col_num_;
	std::vector<std::vector<NodePtr>> nodes_;
	
	FRIEND_WITH_LAYER;
private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new HoneycamLayer()); }

public:
	SOMLayer(uint row_num, uint col_num) : row_num_(row_num), col_num_(col_num),
		nodes_(std:vector<std::vector<NodePtr>>(row_num, std::vector<NodePtr>(col_num)))
	{
		for(uint r=0; r<row_num; ++r){
			for(uint c=0; c<col_num; ++c){
				nodes_[r][c]->AddInEdge();
			}
		}
	}
};

/*
class HoneycamLayer : public SOMLayer
{
	const uint node_num_;
	std::vector<NodePtr> nodes_;
	
	FRIEND_WITH_LAYER;
private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new HoneycamLayer()); }

public:
	HoneycamLayer(uint row_num, uint col_num) : Layer(row_num * col_num){}
};
*/
template <class InputInfo_, class OutputInfo_>
class SOM_Online
{
public:
	using DataFormat = DataFormat<InputInfo_, OutputInfo_>;

private: 
	
private:
	void Init();
	
public:
	SOM_Online(){}
	
	void Train(InputDataPtr input);
};

template <class InputInfo_, class OutputInfo_>
void SOM_Online<InputInfo_, OutputInfo_>::Init()
{
}

}
#endif