NeuralNetwork (Implemented in C++11)
=============
現在まったり作成中 - Now Implementing -  
確認済み動作環境：Visual Studio 2013 RC

【実装済み (ちゃんと動くもの)】  
* Multi-Layer Perceptron (online training)  
* Multi-Layer Perceptron (batch training)　※性能微妙  
* Auto-Encoder (online training)  

【実装・テスト中】  
* Auto-Encoder (batch training)  
* Deep Auto-Encoder  

【変更・修正メモ】  
todo:バッチ学習の性能向上方法のサーベイ, 検証結果のまとめ  

2014/1/4
・重み更新時のエッジ重み減衰によるL2正則化の実装・検証

2013/12/30
・訓練データ作成クラスによる入力データの形式を規定

2013/12/25
・バッチ学習の並列化実装
