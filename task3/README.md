# CORSMAL2021

## Task3: Mask R-CNN → LoDE

* LoDEを一部修正して使用.

* segmentation後, 手の凹みが残る.
  + 左右のうちどちらかは必ず手の表面積が少ないので, 左右のview(=view{1,2}) を使用する.
  + サンプリング点群を2次元に投影して, maskの内側かどうか判定するとき, view-1とview-2とでmaskの論理積を取るのではなく, 論理和を取る. (シリンダー=回転体で近似するのでこっちのほうが合理的な気がする)

* 点群各Q_iについて, Q_iとQ_{i+1}とを三角錐を潰した形で近似して積分チックに計算する.

* 1fps分ずつフレームを精査して, 以下に示すスコアリングをもとに動的にフレームを決定. ただし, 最初と最後のフレームは必ず使う.
  + すなわち, segmentationで得られたmaskの総pixel数から, 上位3つのBBOXの重なりを引いたものをスコアとして, スコアが最も高いフレームを選択.

* ピッチャーを取ってこないように, BBOX上位3つに対して, 左から右に物体検出のスコアに重み付けをしてあげる
  + あんまりよろしくないかもしれないので, もしかしたら消すかも


## Score

### Validation dataset (80:20, seed=0)

| Container | score | 
| :--- | :---: | 
| Red cup | 0.746 | 
| Small white cup | 0.692 | 
| Small transparent cup | 0.572 | 
| Green glass | 0.658 | 
| Wine glass | 0.672 | 
| Champagne flute glass | 0.455 | 
| Cerela box | 0.661 | 
| Biscuit box | 0.607 | 
| Tea box | 0.718 | 


|  | score | 
| :--- | :---: | 
| Total | 0.640 |
