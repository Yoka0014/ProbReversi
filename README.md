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

C++のコンパイラがインストールされていなかったり、コンパイラへのパスが通っていなかったりすると失敗する。環境に合わせて別途インストールする(Windowsならmsvc、Linux/macOSならgcc など)。

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
6x6盤面の場合

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
6x6盤面の場合

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

player = pos.side_to_move 
opponent = pos.opponent_color 
print(player == DiscColor.WHITE)  # Trueが出力される.
print(opponent == DiscColor.BLACK) # Trueが出力される.
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

### 特定のマス目の着手成功確率を取得する
特定のマス目の着手成功確率はPosition.TRANS_PROBフィールドから取得できる。例えば、座標0の着手成功確率が欲しい場合は、以下のように記述する。

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

print(pos.TRANS_PROB[0])
```

### 等価演算子による盤面の比較
Positionオブジェクトは等価演算子で比較可能。手番(side_to_move)が同じ色で、かつ、盤上の石の配置が等しく、かつ、各マス着手成功確率が等しければ、同じオブジェクトと判定される。

### Positionオブジェクトのコピー
Position.copyメソッドを用いると、オブジェクトをコピーできる。 既存のPositionオブジェクトに別のPositionオブジェクトの内容をコピーしたい場合は、Position.copy_toメソッドを用いる。  
この際、着手成功確率のリスト(Position.TRANS_PROB)をコピーしたくない場合は、copy_trans_probにFalseを指定する。

```python
from prob_reversi import Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

pos_1 = pos.copy()
print(pos == pos_1)  # データが同一なのでTrue.
print(pos is pos_1)  # 異なるインスタンスなのでFalse.

pos_2 = Position(6)
pos.copy_to(pos_2)  # pos_2にposの内容をコピー.
print(pos == pos_2)  # データが同一なのでTrue.
print(pos is pos_2)  # 異なるインスタンスなのでFalse.

pos_2 = Position(6)
pos.copy_to(pos_2, copy_trans_prob=False)  # pos_2にposの内容をコピー. ただし, 着手成功確率はコピーしない.
print(pos == pos_2)  # 着手成功確率が異なるのでFalse.
print(pos is pos_2)  # 異なるインスタンスなのでFalse.

```

### 盤面の編集
盤面の初期化など、他の石を裏返さずに石を配置したい場合は、Position.put_player_disc_at、Position.put_opponent_disc_atメソッドを用いる。前者は現在の手番の石を、後者は相手の石を配置する。  
特定のマス目にある石を取り除く場合には、Position.remove_disc_atメソッドを用いる。

### 終局判定
Position.is_gameoverメソッドで終局かどうか判定できる。ただし、探索などで僅かでも速度を求めるなら、このメソッドは使わずに「パスが2回連続発生したら終局」という判定方法の方が高速。

### 石差の取得
現在の手番側からみた石差を取得したい場合は、Position.get_scoreメソッドを用いる。 例えば、手番の石が33個、相手の石が4個ある場合、 $33 - 4 = 28$ がPosition.get_scoreメソッドの戻り値となる。

### パスが可能な局面かどうか判定する
リバーシでは、手番のプレイヤーが置ける場所がない場合、パスをして相手に手番を譲る。パスが可能かどうかの判定はPosition.can_passメソッドで行う。Position.can_passメソッドは、パスが可能ならTrue、そうでなければFalseを返す。  
後述のPosition.get_next_movesメソッドと併用する場合は、わざわざPosition.can_passメソッドでパスが可能か判定するよりも、Position.get_next_movesメソッドが1つも着手可能位置を返してこなかったらパスが可能であると判定したほうが速い。

### 合法手かどうか確認する
Position.is_legalメソッドに着手位置の座標を渡すことで、その位置がルール上着手可能な場所かどうか確認できる。

```python
from prob_reversi import Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

print(pos.is_legal(8))  # 初期局面では座標8に着手可能なのでTrueが出力される.
print(pos.is_legal(0))  # 初期局面では座標0に着手はできないのでFalseが出力される.
```

### パスをする
Position.do_passメソッドを用いれば、着手を行わずに相手に手番を譲ることができる。このメソッドはPosition.can_passメソッドの戻り値の真偽に関わらず利用できる。つまり、ルール上パスできない局面もパスできる。

### 着手可能位置を取得する
Position.get_next_movesメソッドを用いると、現在の手番のプレイヤーが着手可能な場所の座標を取得できる。Position.get_next_movesメソッドはint型のGeneratorを返すので、着手を全て保持したい場合は、リストなどに格納する必要がある。

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

for coord in pos.get_next_moves():  # pos.get_next_movesはGeneratorを返すので, forで列挙すればリストなどに格納するより高速.
    print(pos.convert_coord_to_str(coord))

move_list = list(pos.get_next_moves())  # 着手可能位置を全て保持したいのであれば, Generatorからリストを作ればよい.  
print(move_list)    

"""
出力結果

C2
B3
E4
D5
[8, 13, 22, 27]
"""
```

### 着手可能位置を1つだけサンプリングする
Position.sample_next_moveメソッドを用いれば、疑似乱数を用いて着手可能位置から1つだけランダムに取得できる。

### 着手位置から実際の着手を生成する
Position.get_moveメソッドに着手する座標を与えれば、着手を表現するMoveオブジェクトを取得できる。Moveクラスは以下のように定義されている。

```Python
class Move:
    """
    着手を表現するクラス.

    Attributes
    ----------
    player: Player
        どちらのプレイヤーの石が着手されるか.
    coord: int
        着手位置の座標.
    flip: int
        裏返る石の配置.
    """
    def __init__(self, player=Player.NULL, coord=0, flip=0):
        self.player = player
        self.coord = coord   
        self.flip = flip
```

Position.get_moveメソッドから取得できるMoveオブジェクトは、着手するマスの着手成功確率によって変化する。つまり、同じ状態のPositionオブジェクトであっても、呼び出すたびにPosition.get_moveメソッドの戻り値は異なることがある。  
もし、着手成功確率に関わらず、着手に成功した場合のMoveオブジェクトと着手に失敗した場合のMoveオブジェクトが欲しい場合は、Position.get_player_moveメソッドとPosition.get_opponent_moveメソッドを用いればよい。

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

coord = 8   # 座標8に着手したい.
prob_move = pos.get_move(coord)     # posの各マス目の着手成功確率は全て0.5に設定しているので, prob_moveは0.5の確率で着手に成功した場合のMoveオブジェクトになる.
success_move = pos.get_player_move(coord)   # 確実に着手に成功した場合のMoveオブジェクトを取得する.
failure_move = pos.get_opponent_move(coord) # 確実に着手に失敗した場合のMoveオブジェクトを取得する.
```

### Moveオブジェクトを用いて着手する
Position.do_moveメソッドにMoveオブジェクトを渡せば、着手が完了し、手番が入れ替わる。ただし、Position.do_moveメソッドは合法手判定を一切行わないため、不正なMoveオブジェクトを渡すと、盤面情報が壊れることに注意。もし、合法手判定を行ったうえで着手をする場合は、後述のPosition.do_move_atメソッド用いること。


```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

coord = 8   # 座標8に着手したい.
prob_move = pos.get_move(coord)     # posの各マス目の着手成功確率は全て0.5に設定しているので, prob_moveは0.5の確率で着手に成功した場合のMoveオブジェクトになる.
success_move = pos.get_player_move(coord)   # 確実に着手に成功した場合のMoveオブジェクトを取得する.
failure_move = pos.get_opponent_move(coord) # 確実に着手に失敗した場合のMoveオブジェクトを取得する.

p = pos.copy()
p.do_move(prob_move)
print(p)    # どんな盤面が出力されるかは, prob_moveの中身次第.

print()

pos.copy_to(p, copy_trans_prob=False)
p.do_move(success_move)
print(p)    # 着手に成功した場合の盤面が出力される.

print()

pos.copy_to(p, copy_trans_prob=False)
p.do_move(failure_move)
print(p)    # 着手に失敗した場合の盤面が出力される.
```

### 座標情報のみで着手する
Position.get_moveメソッドで着手を取得せずに、座標情報のみで着手を行いたい場合は、Position.do_move_atメソッドを用いる。Position.do_move_atメソッドは、着手成功確率に従って確率的に着手をする。また、このメソッドは合法手判定を行うため、非合法手が渡された場合は、盤面を更新せずにFalseを返す。

```python
from prob_reversi import DiscColor, Position

pos = Position(6, [0.5 for _ in range(6 * 6)])

coord = 8   # 座標8に着手したい.
print(pos.do_move_at(coord))    # 合法手なのでTrueを出力.
print(pos)  # どんな盤面が出力されるかは, pos.do_move_atの成否次第.

print(pos.do_move_at(0))    # この時点で座標0に石は置けないのでFalse
```

### Position.do_moveメソッドとPosition.do_move_atメソッドの使い分け
Position.do_move_atメソッドは合法手判定を行う分、速度が遅いので、探索などで用いる場合は、Position.do_moveを用いるべき。なぜなら、探索プログラムが正しく実装されていれば、Position.do_moveに非合法手が渡されるはずがないから(もちろんデバッグ時は合法手判定を挟んだほうがよいかもしれない)。  
UIからユーザーの着手を受け付ける場合は、Position.do_move_atメソッド用いるほうが適切。ユーザーは誤って非合法手を入力してしまう場合があるので。

## オリジナルのプレイヤーを作る

### プレイヤーのインターフェース
オリジナルのプレイヤー(エージェント)はクラスの形で実装する。あらゆるプレイヤークラスは、game.pyで定義されているIPlayerインターフェースを実装する必要がある。このインターフェースを
実装することで、異なる仕組みで動作するプレイヤー同士を対戦させることができる。IPlayerは以下のように定義されている。

```python
from abc import ABCMeta, abstractproperty, abstractmethod  # 抽象クラスを定義するために必要.
from prob_reversi import Move, Position

class IPlayer(metaclass=ABCMeta):
    """
    プレイヤーが実装するインターフェース.
    """
    @abstractproperty
    def name(self) -> str:
        """
        プレイヤーの名前.
        """
        pass

    @abstractmethod
    def set_position(self, pos: Position):
        """
        局面を設定する.
        """
        pass

    @abstractmethod
    def gen_move(self) -> int:
        """
        着手を決定して返す.
        """
        pass

    @abstractmethod
    def do_move(self, move: Move):
        """
        与えられた着手で局面を更新する.
        """
        pass

    @abstractmethod
    def do_pass(self):
        """
        パスをする.
        """
        pass
```

### プレイヤークラスを定義する
ここでは、新たに作るプレイヤーの名前をMyPlayerとする。自作のプレイヤーには以下のようにIPlayerクラスを継承させる。

```python
from game import IPlayer
from prob_reversi import Move, Position

class MyPlayer(IPlayer):
    def __init__(self):
        self.__pos = Position(4)
```

### プレイヤーの名前を設定する
以下、クラスの宣言部分は省略して記述する。
プレイヤーの名前は、IPlayer.nameメソッドを定義して、それを利用して名前を返す。nameメソッドには@property属性を付ける。

```python
@property
def name(self):
    return "My Player"
```

### 盤面を設定する
対局開始時に、現在の盤面が後述のGameクラスによって通知される。その際に呼び出されるのがIPlayer.set_positionメソッド。set_positionメソッドでは、現在の盤面を設定すると共に必要であれば、プレイヤーの初期化を行う。以下のコードはset_positionメソッドの最低限の実装。

```python
def set_position(self, pos: Position):
    self.__pos = pos
```

### 着手を決定する
対局中に手番になると、Gameクラスからgen_moveメソッドが呼び出される。このメソッドでは次に着手する場所の座標を戻り値で返す。ただし、盤面の更新はこの時点では行わず、後述のIPlayer.do_moveメソッドで行うこと。以下のコードは、着手成功確率が最も高い場所に着手する実装。

```python
def gen_move(self) -> int:
    pos = self.__pos
    moves = pos.get_next_moves()
    return max(moves, lambda c: pos.TRANS_PROB[c])
```

### 盤面を更新する
どちらか一方のプレイヤーが着手した後、Gameクラスによって両プレイヤーのIPlayer.do_moveメソッドが呼び出されることで、両プレイヤーが保持する盤面が更新される。
以下のコードはdo_moveメソッドの最低限の実装。

```python
def do_move(self, move: Move):
    self.__pos.do_move(move)
```

### パスの時
現在のプレイヤーに石を置ける場所がない場合は、Gameクラスによって両プレイヤーのIPlayer.do_passメソッドが呼び出される。
以下のコードはdo_passメソッドの最低限の実装。

```python
def do_pass(self):
    self.__pos.do_pass()
```

## プレイヤーの作り方
プレイヤーはクラスとして実装して，そのクラスはIPlayerインターフェースを実装する． 
自作するプレイヤーをMyPlayerとして以下説明する．

```python
from game import IPlayer
from prob_reversi import Move, Position

class MyPlayer(IPlayer):
    def __init__(self):
        self.__pos = Position(4)
    
    @property
    def name(self):
        return "My Player"

    def set_position(self, pos: Position):
        self.__pos = pos
    
    def gen_move(self) -> int:
        pos = self.__pos
        moves = pos.get_next_moves()
        return max(moves, lambda c: pos.TRANS_PROB[c])
    
    def do_move(self, move: Move):
        self.__pos.do_move(move)

    def do_pass(self):
        self.__pos.do_pass()

```

### コンストラクタ
コンストラクタでは，プレイヤーが内部で保有する値を初期化する．  
MyPlayerの場合は，盤面のみを保有するので，盤面の初期化のみを行う.

### nameプロパティ
自分のプレイヤーの名前を返せばよい．

### set_position関数
新しい盤面を設定するときに呼び出される. 
新しい盤面がセットされた際に，他に何かやりたい処理があれば適宜この関数内に書く．　

### gen_move関数
次の着手を決定する関数．現在の盤面から，次の手を決定して，その座標を返せばよい．　　
例としてMyPlayerでは，着手成功確率が最も高い場所に着手するように実装している．

### do_move関数
この関数が呼び出されたとき，引数で与えられたMoveオブジェクトに基づいて，自身が保有するPositionオブジェクトを更新する. 
盤面の更新の際に，ほかに何かやりたい処理があれば，この関数内に書く．

### do_pass関数
do_move関数のパスバージョン． 
相手がパスしたときは，これが呼び出される.


## オリジナルのプレイヤーを対戦させる
プレイヤー同士の対戦は、Gameクラス(game.py)によって管理される。具体的な対局の流れは以下の通り。

![対局の流れ](https://user-images.githubusercontent.com/128400304/232414781-2efc73a2-d075-4785-83fa-c92bec516d1b.jpg)

### 対戦のさせ方



