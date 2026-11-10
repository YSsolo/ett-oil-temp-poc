"""PoC 報告スライド（PDF）を、results/metrics.json と figs/*.png から組み立てる。

「メッセージ & ボディ」構成: 各スライドの上段に結論の 1 文（メッセージ）、下段に根拠（図・表・箇条書き）。
実行: python src/build_slides.py  （先に src/poc.py を実行）
フォント: 環境変数 SLIDE_FONT で日本語 TTF を指定（既定は Windows の游ゴシック/メイリオ）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
FIGS, RES = ROOT / "figs", ROOT / "results"
OUT = ROOT / "PoC_report_ETT_oil_temperature.pdf"

cands = [os.environ.get("SLIDE_FONT", ""), r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\msgothic.ttc"]
FONT = next((c for c in cands if c and Path(c).exists()), None)
if FONT is None:
    raise SystemExit("日本語フォントが見つかりません。SLIDE_FONT に TTF/TTC のパスを指定してください。")
pdfmetrics.registerFont(TTFont("JP", FONT, subfontIndex=0))

W, H = 960, 540
NAVY, ORANGE, GRAY, LIGHT = colors.HexColor("#1f3a5f"), colors.HexColor("#e07b00"), colors.HexColor("#6b7280"), colors.HexColor("#eef2f7")
M = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
E1, E2 = M["ETTh1"], M["ETTh2"]


def S(size, leading=None, color=colors.black, **kw):
    return ParagraphStyle("s", fontName="JP", fontSize=size, leading=leading or size * 1.45, textColor=color, **kw)


def para(c, text, x, y_top, w, style):
    p = Paragraph(text, style)
    _, h = p.wrap(w, 1000)
    p.drawOn(c, x, y_top - h)
    return h


def mae(res, h, k):
    return res[f"h{h}"][k]["MAE"]


def key(res, prefix):
    return next(k for k in res["h1"] if k.startswith(prefix))


K_ONLY, K_LOAD, K_RIDGE = key(E1, "lgb_OTの"), key(E1, "lgb_OT+"), key(E1, "ridge")
gain = {h: (1 - mae(E1, h, K_ONLY) / mae(E1, h, "persistence")) * 100 for h in [1, 6, 24, 48]}
gain2 = {h: (1 - mae(E2, h, K_ONLY) / mae(E2, h, "persistence")) * 100 for h in [1, 6, 24, 48]}

pages = []


def slide(message, body, no_total=True):
    pages.append((message, body))


def header(c, message, n):
    c.setFillColor(NAVY)
    c.rect(0, H - 92, W, 92, stroke=0, fill=1)
    para(c, message, 36, H - 18, W - 72, S(21, 30, colors.white))
    c.setFillColor(GRAY)
    c.setFont("JP", 9)
    c.drawString(36, 16, "ETT 油温予測 PoC 報告")
    c.drawRightString(W - 36, 16, str(n))


def img(c, name, x, y, w=None, h=None):
    from reportlab.lib.utils import ImageReader
    ir = ImageReader(str(FIGS / name))
    iw, ih = ir.getSize()
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    c.drawImage(ir, x, y, width=w, height=h, mask="auto")


def table(c, data, x, y_top, col_w, font=11, header_bg=NAVY, row_h=24):
    rows = [[Paragraph(str(v), S(font, font * 1.25, colors.white if i == 0 else colors.black)) for v in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=col_w, rowHeights=[row_h] * len(rows))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
    ]))
    w, h = t.wrap(0, 0)
    t.drawOn(c, x, y_top - h)
    return h


def bullets(c, items, x, y_top, w, size=13, gap=7):
    y = y_top
    for it in items:
        h = para(c, "・" + it, x, y, w, S(size, size * 1.5))
        y -= h + gap
    return y_top - y


# ------------------------------------------------------------------ スライド定義
def s1(c):
    bullets(c, [
        f"<b>6時間先の油温は、現在値をそのまま使う方法より誤差を {gain[6]:.1f}% 減らせた</b>（MAE {mae(E1,6,'persistence'):.2f} → {mae(E1,6,K_ONLY):.2f} ℃、ETTh1 テスト）。48時間先でも {gain[48]:.1f}% 減（{mae(E1,48,'persistence'):.2f} → {mae(E1,48,K_ONLY):.2f} ℃）。",
        f"<b>ただし改善幅は小さい</b>。1時間先は {gain[1]:.1f}% 減（0.42 → 0.41 ℃）、24時間先は {gain[24]:.1f}% 減にとどまる。油温は直近値の影響が強く、現在値の持ち越しがすでに強い基準になっている。",
        "<b>負荷6変数（HUFL〜LULL）を足しても精度は上がらなかった</b>。油温の履歴だけのモデルが、全ホライズンで最良か同等だった。",
        "<b>このデータだけで予防保全の予測に使うのは時期尚早</b>。学習期間（夏〜秋）とテスト期間（冬）で油温の水準が大きく違い、外気温など季節を説明するデータがない。",
        "提案: ①外気温・天候データの追加、②1〜6時間先の短期アラートから導入、③長期の誤差が許容できるかを運用側の要件で確認。",
    ], 40, H - 125, W - 80, size=15, gap=11)


def s2(c):
    bullets(c, [
        "<b>問い</b>: 油温（OT）の将来値を予測すると、運用・保全の判断にどの程度価値が出るか。",
        "<b>データ</b>: ETTh1（主）と ETTh2（汎化の確認）。1時間間隔、17,420行（2016-07〜2018-06）。目的変数は OT、説明変数は負荷6変数（HUFL, HULL, MUFL, MULL, LUFL, LULL）。欠損なし。",
        "<b>分割</b>: ETT の標準どおり時系列順に 学習12か月 / 検証4か月 / テスト4か月。テスト期間は学習・検証・特徴量の作成・ハイパーパラメータ選択のどこにも使っていない。",
        "<b>予測先</b>: 1・6・24・48時間先。時刻 T の特徴量（T の負荷を含む）で T+h の OT を予測する（課題文の「t=T の特徴量は使用してよい」に従う）。",
        "<b>評価</b>: MAE・RMSE（℃）。基準は、①現在値をそのまま使う（persistence）、②24時間前の同時刻の値を使う。",
        "<b>手法</b>: Ridge と LightGBM。油温履歴のみ / 油温＋負荷 の2通りで、負荷の効果を切り分ける。",
    ], 40, H - 125, W - 80, size=14, gap=10)


def s3(c):
    d = M["eda"]["ETTh1"]
    img(c, "01_ot_overview.png", 36, H - 360, w=560)
    data = [["期間", "平均 OT", "標準偏差"]]
    for k, lab in [("train", "学習"), ("val", "検証"), ("test", "テスト")]:
        data.append([lab, f"{d['OT_mean_by_split'][k]:.1f} ℃", f"{d['OT_std_by_split'][k]:.1f} ℃"])
    table(c, data, 620, H - 120, [90, 100, 100], font=12)
    bullets(c, [
        "水準が期間で大きく動く。学習の平均 17.1 ℃ に対し、テスト（2017-11〜2018-02）は 4.8 ℃。",
        "2016年7月の高温（最大約46 ℃）は学習の序盤だけにある。",
        "→ 絶対値を覚えるモデルはテストで外れる。<b>水準に依存しない設計</b>が要る。",
    ], 620, H - 230, 310, size=12.5, gap=8)
    para(c, "※ 2018-03以降の約4か月は、標準の12/4/4か月の分割に含まれないため未使用。", 36, 70, 560, S(10, 14, GRAY))


def s4(c):
    img(c, "02_seasonality.png", 30, H - 280, w=520)
    img(c, "04_lag_corr.png", 560, H - 330, w=360)
    cr = M["eda"]["ETTh1"]["corr_with_OT"]
    bullets(c, [
        "<b>日内の周期はあるが小さく、水準の差は季節が支配的</b>。時刻別の平均は 6時の12.3 ℃ から16時の14.7 ℃ までの約2 ℃ の振れ。一方、月平均は 2016年7月の約34 ℃ から 2018年1月の約2 ℃ まで動く。",
        f"<b>負荷との相関は弱い</b>。OT と最も相関が高いのは HULL（{cr['HULL']:.2f}）と MULL（{cr['MULL']:.2f}）。HUFL は {cr['HUFL']:.2f}。",
        "負荷の遅れ相関（右図）は、HULL・MULL で遅れ10時間前後に0.26まで上がる程度で、強くはない。HUFL は24時間周期の波（最大0.15）。負荷の履歴は、油温の履歴以上の情報をあまり足さないと予想できる。",
    ], 40, H - 305, W - 80, size=12.5, gap=7)


def s5(c):
    bullets(c, [
        "<b>目的変数は「現在の油温からの変化量」</b>（OT[T+h] − OT[T]）。予測後に現在値を足し戻す。水準の差を吸収できる。",
        "<b>特徴量は水準に依存しない形</b>: 過去の油温は「現在値との差」、移動平均も「現在値との差」、変動（標準偏差）、直近1・6時間の変化。年周期（季節）の特徴量は入れない（学習が1.5年分で、季節を覚えてしまうため）。",
        "<b>負荷6変数</b>はT時点の値、1・3・6・24時間前の値、24時間平均。時刻と曜日は周期特徴量にする。",
        "<b>モデル</b>: LightGBM（木15葉、最小80サンプル、学習率0.03）。検証期間で早期終了して木の本数を決め、その本数で学習+検証の全期間を使って学習し直す。テストには触れない。Ridge（α=100）も比較。",
        "<b>設計上の意思決定</b>: 最初は絶対値の特徴量で作ったところ、24時間先以降で現在値のままより悪化した（MAE 2.16 ℃ 対 1.53 ℃）。水準依存が原因と考え、上の設計に変えて解消した。",
    ], 40, H - 125, W - 80, size=13.5, gap=10)


def s6(c):
    img(c, "06_mae_by_horizon.png", 30, H - 400, w=470)
    data = [["予測先", "現在値のまま", "LightGBM（OTのみ）", "改善率"]]
    for h in [1, 6, 24, 48]:
        data.append([f"{h}h", f"{mae(E1,h,'persistence'):.3f}", f"{mae(E1,h,K_ONLY):.3f}", f"{gain[h]:.1f}%"])
    th = table(c, data, 520, H - 120, [70, 100, 140, 70], font=11.5)
    bullets(c, [
        "テスト期間（2017-11〜2018-02）の MAE [℃]。LightGBM は全ホライズンで基準以下。",
        "予測先が遠いほど誤差が増える。48時間先で 1.8 ℃。",
        "6時間先と48時間先の改善が相対的に大きい（9.5%、7.9%）。",
        "Ridge は基準とほぼ同等。非線形（負荷が高いときの昇温の傾き、時刻との相互作用）を拾う余地は小さい。",
    ], 520, H - 120 - th - 14, 410, size=12, gap=6)


def s7(c):
    img(c, "05_pred_vs_true_h24.png", 30, H - 290, w=640)
    img(c, "07_importance.png", 680, H - 330, w=250)
    bullets(c, [
        "24時間先の予測（橙）は、現在値のまま（灰の点線）より谷や山のずれが少し小さい。ただし1月上旬の急な冷え込み（-2 ℃ 付近）は両方とも遅れる。",
        "重要度の上位は、1週間平均との差、1週間前との差、1週間の変動幅。次に HULL の24時間平均、曜日。1週間スケールの水準の位置づけを、モデルが最も使っている。",
    ], 40, H - 320, 620, size=12.5, gap=8)


def s8(c):
    def row(res, g, name):
        return [name] + [f"{mae(res,h,K_ONLY):.2f} / {mae(res,h,K_LOAD):.2f}" for h in [1, 6, 24, 48]]
    data = [["MAE [℃]（OTのみ / OT＋負荷）", "1h", "6h", "24h", "48h"], row(E1, gain, "ETTh1"), row(E2, gain2, "ETTh2")]
    table(c, data, 40, H - 118, [250, 120, 120, 120, 120], font=12, row_h=28)
    data2 = [["現在値のまま（基準）MAE [℃]", "1h", "6h", "24h", "48h"],
             ["ETTh1"] + [f"{mae(E1,h,'persistence'):.2f}" for h in [1, 6, 24, 48]],
             ["ETTh2"] + [f"{mae(E2,h,'persistence'):.2f}" for h in [1, 6, 24, 48]]]
    table(c, data2, 40, H - 225, [250, 120, 120, 120, 120], font=12, row_h=28)
    bullets(c, [
        "<b>負荷を足すと、むしろ僅かに悪化する</b>（ETTh1 の48時間先で 1.78 → 1.90 ℃）。油温の履歴にすでに負荷の影響が含まれ、残る情報は少ない。学習が短く、雑音を拾う分が上回ったと考えている。",
        f"<b>ETTh2 でも同じ傾向</b>。ETTh2 の基準は 6時間先で {mae(E2,6,'persistence'):.1f} ℃ と悪く、モデルは大きく改善する（{gain2[6]:.0f}%）。24時間先は {gain2[24]:.0f}%、48時間先は {gain2[48]:.0f}% の改善。",
        "ETTh2 は油温の動きが大きく（標準偏差7〜12 ℃）、現在値の持ち越しが外れやすいデータ。どの変電所でも使えるとまでは、この検証では言えない。",
    ], 40, H - 340, W - 80, size=12.5, gap=9)


def s9(c):
    bullets(c, [
        "<b>限界①</b> 外気温・日射などがない。油温の季節水準は主に外気温で決まるはずで、24時間以上先は精度が頭打ちになる。",
        "<b>限界②</b> 学習が1.5年分しかない。冬の学習データは前年の1回分だけで、季節をまたぐ汎化は確認できていない。",
        "<b>限界③</b> 異常の検知は未評価。温度の「高さ」ではなく「想定より速く上がる」ことを検知できるかは、ラベル（異常記録）がないため評価できていない。",
        "<b>限界④</b> 1時間間隔の ETTh のみ。15分間隔の ETTm は使っていない。",
        "<b>改善方針</b> ①外気温（気象データ。出典を明記して利用）、②複数変圧器の学習、③予測の不確実性（分位点回帰）を出し、閾値を超える確率でアラートにする、④ETTm で短期の応答（負荷変化後の昇温）を詳細に見る。",
        "<b>クライアントへの提案</b> まず1〜6時間先の短期予測を、閾値監視の補助として並行運用し、誤差と有用性を実運用で確認する。長期は外気温の追加後に再検証する。",
    ], 40, H - 125, W - 80, size=13, gap=9)


def s10(c):
    bullets(c, [
        "課題文の通り、不明点は仮定を置いて進めた（質問は受け付けないため）。",
        "<b>仮定1</b> 時刻 T の特徴量（負荷を含む）は T の時点で入手できる。予測は T+h の油温。",
        "<b>仮定2</b> 時刻 T の油温も入手できる（ホライズン1・6・24・48時間）。入手できない場合のナウキャスト（T の負荷と T-1 以前の油温から T の油温を推定）は別途評価し、ETTh1 で MAE 0.41 ℃（T-1 の値を使う方法は 0.42 ℃）だった。",
        "<b>仮定3</b> ETT の標準分割（12/4/4か月、1か月=30日）を使い、再現性のためランダムシードを固定した。",
        "<b>仮定4</b> 追加データは使っていない。出典を明記するデータはなし。",
        "<b>再現手順</b> README を参照（データは zhouhaoyi/ETDataset から取得）。",
    ], 40, H - 125, W - 80, size=13.5, gap=10)


BODY = [
    ("結論: 油温は6時間先で約9.5%、48時間先で約8%誤差を減らせたが、このデータだけで予防保全に使うには足りない", s1),
    ("検証の設定: ETT の標準分割で、テスト期間に一切触れずに「現在値のまま」と比べた", s2),
    ("EDA ①: 油温の水準が期間で大きく動く（学習17.1 ℃、テスト4.8 ℃）ので、絶対値に頼るモデルは外れる", s3),
    ("EDA ②: 日内・季節の周期はあるが、負荷との相関は弱く、負荷の履歴は新しい情報をあまり足さない", s4),
    ("アプローチ: 水準に依存しない「変化量」を学習し、年周期の特徴量を使わない設計に絞った", s5),
    ("結果: LightGBM は全ホライズンで現在値のままを上回るが、改善幅は2〜9.5%と小さい", s6),
    ("予測例と重要度: 1週間スケールの水準（週平均との差）が最も効き、急な冷え込みは予測できない", s7),
    ("負荷6変数は精度を上げず、ETTh2 でも同じ傾向だった", s8),
    ("限界と今後: 外気温の追加と短期アラートからの導入を提案する", s9),
    ("付録: 置いた仮定と再現手順", s10),
]


def main():
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("ETT 油温予測 PoC 報告")
    c.setAuthor("山口 慧")
    # 表紙
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    para(c, "電力用変圧器の油温予測 PoC 報告", 60, H - 170, W - 120, S(34, 46, colors.white))
    para(c, "ETT データセットによる技術検証（EDA、モデル構築、限界と改善方針）", 60, H - 250, W - 120, S(16, 24, colors.HexColor("#cbd5e1")))
    para(c, "山口 慧", 60, 90, 400, S(16, 22, colors.white))
    c.showPage()
    for i, (msg, fn) in enumerate(BODY, start=2):
        c.setFillColor(colors.white)
        c.rect(0, 0, W, H, stroke=0, fill=1)
        header(c, msg, i)
        c.setFillColor(colors.black)
        fn(c)
        c.showPage()
    c.save()
    print("書き出し:", OUT)


if __name__ == "__main__":
    main()
