# QA4U

量子アニーリング（Quantum Annealing）や組合せ最適化問題を、Python で実験するためのサンプル集です。

QUBO 形式で問題を定式化し、OpenJij や D-Wave を用いて解を探索する例を収録しています。あわせて、各テーマの説明用 TeX / PDF ファイルも置いています。

## 内容

| ディレクトリ              | 内容                           |
| ------------------- | ---------------------------- |
| `EightQueen`        | 8クイーン問題                      |
| `KnapSack`          | ナップサック問題                     |
| `MagicCircle`       | 魔方陣・魔円に関する問題                 |
| `NumberPlace`       | ナンバープレース（数独）                 |
| `TravelingSalesman` | 巡回セールスマン問題                   |
| `excel2csv`         | Excel ファイルを CSV に変換する補助スクリプト |

## 必要な環境

Python 3.x

主に以下のライブラリを使用します。

```bash
pip install numpy openjij
```

D-Wave の実機を使う場合は、追加で以下が必要です。

```bash
pip install dwave-system
```

D-Wave を使う場合は、各自の API token を設定してください。

## 実行例

各ディレクトリに移動して Python スクリプトを実行します。

```bash
cd EightQueen
python EightQueen.py
```

```bash
cd KnapSack
python knapsack.py
```

## 各サンプルについて

### EightQueen

8×8 のチェス盤に、互いに取られないように 8 個のクイーンを配置する問題です。

行・列・斜め方向の制約を QUBO として表し、アニーリングによって解を探索します。

### KnapSack

容量制限のあるナップサックに対して、価値が最大になる品物の組み合わせを探索する問題です。

容量制約をペナルティ項として加え、価値最大化を QUBO のエネルギー最小化問題として扱います。

### MagicCircle

魔方陣・魔円のような、数値配置に関する制約充足問題を扱います。

### NumberPlace

ナンバープレース（数独）を制約充足問題として扱うサンプルです。

### TravelingSalesman

複数の都市を一度ずつ訪問し、出発点に戻る経路のうち、総距離が最短となるものを探す巡回セールスマン問題のサンプルです。

### excel2csv

Excel 形式のデータを CSV に変換するための補助ツールです。

## 注意

このリポジトリは、量子アニーリングや QUBO 定式化の学習・実験を目的としたものです。

問題ごとにパラメータやペナルティ係数を変更することで、解の出方や収束の様子を確認できます。

## Author

smat1957

## License

MIT
