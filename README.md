# ETT 油温予測 PoC

電力用変圧器の油温（OT）を、ETT（Electricity Transformer Temperature）データセットで予測し、運用・保全の判断にどの程度価値が出そうかを検証した技術検証です。

- 報告スライド: [`PoC_report_ETT_oil_temperature.pdf`](PoC_report_ETT_oil_temperature.pdf)（10枚、メッセージ & ボディ構成）
- 実装: [`src/poc.py`](src/poc.py)（EDA・モデル・評価・図）、[`src/build_slides.py`](src/build_slides.py)（スライドの組み立て）
- 結果: [`results/metrics.json`](results/metrics.json)、テスト期間の予測値 `results/ETTh1_test_predictions_h*.csv`

## 結論

| 予測先 | 現在値のまま MAE | LightGBM（油温履歴のみ）MAE | 改善率 |
| --- | --- | --- | --- |
| 1時間 | 0.420 | 0.412 | 1.8% |
| 6時間 | 1.070 | 0.969 | 9.5% |
| 24時間 | 1.534 | 1.468 | 4.3% |
| 48時間 | 1.930 | 1.777 | 7.9% |

ETTh1、テスト期間（2017-11〜2018-02）、単位は ℃。

- LightGBM は全ホライズンで「現在値のまま」を上回るが、改善幅は小さい。
- 負荷6変数（HUFL〜LULL）を加えても精度は上がらなかった（油温の履歴だけのモデルが最良か同等）。
- 学習期間（夏〜秋）とテスト期間（冬）で油温の水準が大きく違う（平均 17.1 ℃ 対 4.8 ℃）。そのため、絶対値ではなく「現在値からの変化量」を学習し、年周期の特徴量は使わない設計にした。
- このデータだけで予防保全に使うには足りない。外気温などの追加が必要（詳細はスライド）。

## 設定

- データ: ETTh1（主）、ETTh2（汎化の確認）。1時間間隔。
- 分割: ETT の標準どおり、時系列順に 学習12か月 / 検証4か月 / テスト4か月（1か月=30日）。テスト期間は学習・検証・特徴量・ハイパーパラメータの選択に使っていない。
- 予測: 時刻 T の特徴量（T の負荷を含む）で T+h の OT を予測（h = 1, 6, 24, 48 時間）。
- モデル: LightGBM、Ridge。検証期間で早期終了して木の本数を決め、その本数で学習+検証の全期間を使って学習し直す。
- 基準: 現在値をそのまま使う、24時間前の同時刻の値を使う。

## 再現手順

データは [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) の `ETT-small/ETTh1.csv`、`ETTh2.csv` を `data/` に置きます（このリポジトリには含めていません）。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows。Linux/Mac は .venv/bin/python
.venv/Scripts/python src/poc.py            # results/ と figs/ を作る
.venv/Scripts/python src/build_slides.py   # PoC_report_ETT_oil_temperature.pdf を作る
```

スライドの日本語フォントは、Windows の游ゴシックまたはメイリオを使います。他の環境では環境変数 `SLIDE_FONT` に日本語 TTF のパスを指定してください。

## 置いた仮定

- 時刻 T の特徴量（負荷を含む）と T の油温は、T の時点で入手できる。
- 入手できない場合のナウキャスト（T の負荷と T-1 以前の油温から T の油温を推定）は別途評価した（ETTh1 で MAE 0.41 ℃。T-1 の値を使う方法は 0.42 ℃）。
- 追加データは使っていない。
- 2018-03 以降の約4か月は、標準の分割に含まれないため使っていない。ETTm（15分間隔）も使っていない。
