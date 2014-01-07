NeuralNetwork (Implemented in C++11)
=============
現在まったり作成中 - Now Implementing -  
確認済み動作環境：Visual Studio 2013 RC

【実装済み (ちゃんと動くもの)】  
* Multi-Layer Perceptron (online training)  
* Multi-Layer Perceptron (batch training) 
* Auto-Encoder (online training)  

【実装・テスト中】  
* Auto-Encoder (batch training)  
* Deep Auto-Encoder  

【変更・修正メモ】  
todo:正則化方法・目的関数の種類と特徴のサーベイ, 検証結果のまとめ  

2014/1/8  
・バッチ学習のバグを修正  
・目的関数が最適値となった状態のパラメータを自動記録する機能追加

2014/1/4  
・重み更新時のエッジ重み減衰によるL2正則化の実装・検証

2013/12/30  
・訓練データ作成クラスによる入力データの形式を規定

2013/12/25  
・バッチ学習の並列化実装
