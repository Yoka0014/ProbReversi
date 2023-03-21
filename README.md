# ProbReversi
首藤研 ゲームAIチーム 2024卒学生 新人研修用のプロジェクト。

## 確率リバーシとは
通常のリバーシに着手成功確率という概念を加えたゲーム。各マス目 $n$ に対して着手成功確率 $p_n$ が定められており、確率 $p_n$ で手番の着手が成功し、確率 $(1 - p_n)$ で相手の石が置かれてしまう。  
勝敗の決定方法は通常のリバーシと同様に終局時の石数が多いプレイヤーが勝利、両者の石数が同数の場合は引き分け。 またルール上置ける場所がない場合はパスとなる。

## 概要
ProbReversiでは、確率リバーシの盤面(prob_reversi.Position)とプレイヤー同士の自動対局機能(game.py)を提供する。また、game.IPlayerインターフェースを自作のクラスに実装することで、自作のプレイヤーを追加して対戦させることもできる。  
盤面のサイズは4x4 ~ 8x8まで対応。

## セットアップ
以下、Python3.10のインストールは既に完了しているものとする。

### Cythonのインストール
ProbReversiでは、一部Cythonのコードが含まれるため、インストールが必要。　　
pipを用いて以下のコマンドでインストールする。

```
pip install cython
```

もしくは

```
python3 -m pip install cython
```
環境によっては、python3の部分がpyだったりpythonだったりする。

### コンパイル
ProbReversiはC++のコードも含まれるため、それらのコンパイルが必要になる。  

1. ターミナルを開き、カレントディレクトリをsetup.pyがあるディレクトリに設定する

2. 以下のコマンドを実行する

```
python3 setup.py build_ext --inplace     
```

C++のコンパイラがインストールされていなかったり、コンパイラへのパスが通っていないと失敗する。環境に合わせて別途インストールする(Windowsならmsvc、Linux/macOSならgcc など)。

## prob_reversi.Positionクラスの使い方

### オブジェクト生成
6x6の盤面、各マスの着手成功確率を全て0.5とする場合、以下のようにPositionオブジェクトを生成する。  

```python
from prob_reversi import Position

pos = Position(6, [0.5 for _ in range(6 * 6)])
```

### 盤面の座標の振り方
盤面の座標は以下のように左上から右下にかけて行優先で割り振られている。

```
| 0| 1| 2| 3| 4| 5|
| 6| 7| 8| 9|10|11|
|12|13|14|15|16|17|
|18|19|20|21|22|23|
|24|25|26|27|28|29|
|30|31|32|33|34|35|

PASS_COORD = 36 
```

盤面のサイズをsizeとしたとき、座標(size * size)はパスを表す特殊座標となる。

### 座標の文字列表記
標準入出力などから文字列の形式で座標をやり取りする際は、分かりやすさのためにアルファベットと数字を用いた2次元座標で表現する。
アルファベットは左から水平方向にA, B, C, ... の順で割り振られる。数字は上から下にかけて1, 2, 3, ... の順で割り振られる。
例えば、6x6盤面における座標20は文字列で表記すると"C4"となる(下図参照)。  
パス座標は文字列で表記すると"Pass"となる。

```
    A  B  C  D  E  F
1 | 0| 1| 2| 3| 4| 5|
2 | 6| 7| 8| 9|10|11|
3 |12|13|14|15|16|17|
4 |18|19|20|21|22|23|
5 |24|25|26|27|28|29|
6 |30|31|32|33|34|35|
```

座標文字列 -> 座標 の変換は、Position.parse_coordメソッド  
座標 -> 座標文字列 の変換は、Position.convert_coord_to_strメソッド  
をそれぞれ用いる。

```python
from prob_reversi import Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

print(pos.parse_coord("C4"))  # 出力: 20
print(pos.convert_coord_to_str(20))   # 出力: C4
```

### 盤面を文字列の形式で出力する。
prob_reversi.Positionクラスは__str__メソッドをオーバーライドしているので、print関数を用いれば文字列の形式で出力できる。

```python
from prob_reversi import Position

pos = Position(6, [0.5 for _ in range(6 * 6)])
print(pos)  # 初期盤面が表示される.
```

出力結果
```
  A B C D E F 
1 - - - - - - 
2 - - - - - - 
3 - - O * - - 
4 - - * O - - 
5 - - - - - - 
6 - - - - - - 
```

'*'が黒石、'O'が白石、'-'が空きマスを表す。

### 現在の手番を取得する
Position.side_to_moveプロパティから現在の手番を取得でき、Position.opponent_colorプロパティから相手の石の色を取得できる。 

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])
player = pos.side_to_move # リバーシでは黒が先手なので、player == DiscColor.BLACK
opponent = pos.opponent_color # opponent == DiscColor.WHITE

pos.do_move_at(8) # 1手進めると, pos.side_to_move　と pos.opponent_color の値が入れ替わる.

print(player == DiscColor.WHITE)  # Trueが出力される.
print(opponent == DiscColor.BLACK) # Falseが出力される.
```

### 石数を取得する
Position.player_disc_countプロパティから現在の手番の石数を、Position.opponent_disc_countプロパティから相手の石数を取得できる。  
特定の色の石数が欲しい場合は、Position.get_disc_count_ofメソッドで取得する。  
盤上の全ての石数と空きマスの数はそれぞれ、Position.disc_countプロパティ、Position.empty_square_countプロパティから取得できる。

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

# 座標8(C2)に着手
pos.do_move_at(8)
print(pos)
print()

"""
着手に成功した場合の盤面

  A B C D E F 
1 - - - - - - 
2 - - * - - - 
3 - - * * - - 
4 - - * O - - 
5 - - - - - - 
6 - - - - - - 
"""

"""
着手に失敗した場合の盤面

  A B C D E F 
1 - - - - - -
2 - - O - - -
3 - - O * - -
4 - - * O - -
5 - - - - - -
6 - - - - - -
"""

print(pos.player_disc_count) 
print(pos.opponent_disc_count)  
print(pos.get_disc_count_of(DiscColor.BLACK))
print(pos.get_disc_count_of(DiscColor.WHITE)) 
print(pos.disc_count)  
print(pos.empty_square_count)  

"""
着手に成功した場合の出力

1
4
4
1
5
31
"""

"""
着手に失敗した場合の出力

3
2
2
3
5
31
"""
```

### 特定のマス目にある石を取得する
特定のマス目にある石が手番の石なのか相手の石なのかを取得する場合は、Position.get_square_owner_atメソッドを、特定のマス目にある石の色を取得したい場合は、Position.get_square_color_atメソッド用いる。  
速度の面では、Position.get_square_owner_atメソッドのほうが条件分岐無しでビット演算のみで完結するので高速。

```python
from prob_reversi import DiscColor, Player, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

print(pos.get_square_owner_at(15) == Player.CURRENT)    # 座標15には黒石があり、黒は現在の手番なのでTrue.
print(pos.get_square_owner_at(14) == Player.OPPONENT)    # 座標14には白石があり、白は相手なのでTrue.
print(pos.get_square_owner_at(14) == Player.CURRENT)    # 座標14には白石があるが、白は現在の手番ではないのでFalse.
print(pos.get_square_owner_at(0) == Player.NULL)    # 座標0には石がないのでTrue.

print(pos.get_square_color_at(15) == DiscColor.BLACK)    # 座標15には黒石があるので、True.
print(pos.get_square_color_at(14) == DiscColor.WHITE)    # 座標15には白石があるので、True.
print(pos.get_square_color_at(0) == DiscColor.NULL)    # 座標0には石がないのでTrue.
```

### 等価演算子による盤面の比較
Positionオブジェクトは等価演算子で比較可能。手番(side_to_move)が同じ色で、かつ、盤上の石の配置が等しく、かつ、各マス着手成功確率が等しければ、同じオブジェクトと判定される。

### 盤面の編集
盤面の初期化など、他の石を裏返さずに石を配置したい場合は、Position.put_player_disc_at、Position.put_opponent_disc_atメソッドを用いる。前者は現在の手番の石を、後者は相手の石を配置する。  
特定のマス目にある石を取り除く場合には、Position.remove_disc_atメソッドを用いる。
