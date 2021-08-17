# VoiceConverter
- Cycle GAN系の声質変換モデルの残骸．一応MIT Licenseで公開するので好きに使ってください  
- ノンパラレル，言語情報不使用，少量 (7-8分程度) のデータで美少女声になりたかった

# 学習方法
- それぞれのディレクトリにあるsourceとtargetにそれぞれ変換元の話者，変換先の話者のwave (22.05[kHz]) をぶち込んでtrain.pyを実行  
- 多分config.iniをいじれば，サンプリング周波数は変更できる (多分)  
- モデルを途中から学習させたい場合は，train.pyのCycleGANTrainingの引数model_checkpointを，Noneでなくてモデルのパスにする  
- 学習epochを変更したい場合は，train.pyのnum_epochsを直接変更する  
- infer.pyでモデルを指定して実行すると，sourceの音声が変換されてtarget_testにぶち込まれる  
- メルスペクトログラムを音声に変換するVocoderは，オンラインでNVIDIAが公開しているWaveGlowと，ローカルのVocGANが使用可能 (今だったらHiFi-GANとかの方が良さそう)  
- 切り替えは，infer.pyのvocoderをコメントアウトしたりする  
- VocGANは推論が速い (CPUでリアルタイム推論できそう) けれど，fine-tuneはした方が良い  

# それぞれのモデルの適当な紹介
## CycleGAN-VC3
- NTTの[こちら](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc3/index.html)の論文の実装
- パラメータ数が多いため学習は少し遅いが，言語情報をかなり保持したまま声質の変換が可能な印象
- 声質の変化と言語情報の保持のバランスが良い感じ

## MaskCycleGAN-VC
- これもNTTの[こちら](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/maskcyclegan-vc/index.html)の論文の実装
- 言語情報はかなり保持されるが，声質が少し元の特徴が残る印象
- また，韻律に関しても，CycleGAN-VC3より不安定な印象
- 今回の実装では，Maskのパラメータは，評価が最も良かった50%ものを使用

## Syclone
- 東北大の[こちら](http://www.spcom.ecei.tohoku.ac.jp/nose/research/scyclone_202001/)の論文の実装
- 原著では通常のスペクトログラムの予測だが，Vocoderの学習が手間だったのでメルスペクトログラムの予測とした
- 私が行った学習は，総音声データが8分未満とかなり少なかったためMode Collapseが発生した
- 正規化手法をSpectral NormalizationからInstance Normalizationへ変更したところ，Mode Collapseの発生は無くなったが，韻律情報が不安定になった
  - 女性声優の演技と，男性の話し声の調子では，その韻律分布がかなり異なるっぽいので，スタイル変換という目的からしたら当然と言えば当然の結果

## UGATIT-VC
- Selfie2Animeのタスクで使用される[U-GAT-IT](https://github.com/taki0112/UGATIT)を声質変換用に魔改造したヤツ，「現実的な声→アニメ声なので似たタスクでは？？」と思い実装
- 45-50k[itr]くらいで，それなりに良い感じになった
- しかし，学習を進めると言語情報が保持できなくなった
- Skip Connectionを入れるとマシになったが，それでもやはり収束先は言語ではない
- 声質はかなり変わったが，やはり声質の変換と言語情報はトレードオフの関係になるらしい

## ANU-Net-VC
- UGATIT-VCが思ったよりいい感じなので，「Attentionって良いのでは？？」と思い[Attention-based nested U-Net](https://www.sciencedirect.com/science/article/abs/pii/S0097849320300546)を実装
- 全然ダメだった，
  - 元々Semantic Segmentation用のモデルなので当たり前では？？

# Next-ToDo?
- Attentionを使用すれば声質がかなり変換され，Maskをすれば言語情報が保持される → 両方合わせれば最強なのでは？？？
  - ~~BERTかな？？~~
