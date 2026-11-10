"""ETT(Electricity Transformer Temperature) 油温(OT)予測 PoC.

ETTh1 を主データ、ETTh2 を汎化の確認に使う。分割は ETT の標準（時系列順に 12 / 4 / 4 か月）。
実行: python src/poc.py  （data/ に ETTh1.csv, ETTh2.csv が必要。入手先は README 参照）
出力: results/metrics.json, results/*.csv, figs/*.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS, RES = ROOT / "data", ROOT / "figs", ROOT / "results"
FIGS.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

LOADS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
HORIZONS = [1, 6, 24, 48]  # 時間先
SEED = 0
BLUE, ORANGE, GRAY, GREEN, RED = "#1f4e8c", "#e07b00", "#8a8f98", "#2a8f5b", "#c0392b"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv", parse_dates=["date"]).set_index("date")
    return df


def split_bounds(n: int):
    """ETT 標準: 12/4/4 か月（1 か月 = 30 日 = 720 時間として 8640 / 2880 / 2880）。"""
    tr, va, te = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    return tr, tr + va, tr + va + te


# ---------------------------------------------------------------- 特徴量
def make_features(df: pd.DataFrame, h: int, use_loads: bool = True, nowcast: bool = False):
    """時刻 T の特徴量で T+h の OT を予測する。T 時点の特徴量（負荷を含む）は使ってよい（課題文）。

    nowcast=True のときは h=0（T の油温そのものを、T の負荷と T-1 以前の油温から推定）。
    """
    ot = df["OT"]
    f = {}
    first_lag = 1 if nowcast else 0
    base = ot.shift(first_lag)  # 基準となる直近の油温（絶対値は特徴量に入れず、基準との差だけを使う）
    # 理由: 学習期間（夏〜秋）と評価期間（冬）で油温の水準が大きく違う。木モデルは学習範囲外の
    # 絶対値を外挿できないので、「基準からの差」と「基準への変化量」で学習させる。
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        if lag > first_lag:
            f[f"OT_rel_lag{lag}"] = ot.shift(lag) - base
    for w in [6, 24, 168]:
        f[f"OT_rel_mean{w}"] = base.rolling(w).mean() - base
        f[f"OT_std{w}"] = base.rolling(w).std()
    f["OT_diff1"] = base.diff(1)
    f["OT_diff6"] = base.diff(6)
    if use_loads:
        for c in LOADS:
            f[c] = df[c]
            for lag in [1, 3, 6, 24]:
                f[f"{c}_lag{lag}"] = df[c].shift(lag)
            f[f"{c}_mean24"] = df[c].rolling(24).mean()
        f["load_sum"] = df[["HUFL", "MUFL", "LUFL"]].sum(axis=1)
    hour = df.index.hour + df.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow = df.index.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    # 季節（年周期）の特徴量は入れない: 学習データが 1.5 年しかなく、季節の水準が過学習の原因になる。
    X = pd.DataFrame(f, index=df.index)
    y = ot.shift(-h)  # h=0 のとき y = OT_T
    return X, y


def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2)))}


def run_one(df, h, use_loads=True, nowcast=False, model="lgb"):
    X, y = make_features(df, h, use_loads, nowcast)
    n = len(df)
    a, b, c = split_bounds(n)
    valid = X.notna().all(axis=1) & y.notna()
    idx = np.arange(n)
    tr = valid.values & (idx < a)
    va = valid.values & (idx >= a) & (idx < b)
    te = valid.values & (idx >= b) & (idx < c)
    # 目的変数は「現在(または直近)の OT からの差分」で学習し、後で足し戻す
    anchor = df["OT"].shift(1 if nowcast else 0)
    ytr, yva, yte = (y - anchor)[tr], (y - anchor)[va], (y - anchor)[te]
    trva = tr | va
    if model == "ridge":
        sc = StandardScaler().fit(X[trva])
        m = Ridge(alpha=100.0).fit(sc.transform(X[trva]), (y - anchor)[trva])
        pred = m.predict(sc.transform(X[te]))
        imp = None
    else:
        params = dict(learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
        # 1) 検証期間で早期終了して木の本数を決める
        m = lgb.LGBMRegressor(n_estimators=2000, **params)
        m.fit(X[tr], ytr, eval_set=[(X[va], yva)], eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        # 2) 決めた本数で、学習+検証の全期間で学習し直す（テスト期間は一切使わない）
        m = lgb.LGBMRegressor(n_estimators=max(m.best_iteration_, 50), **params)
        m.fit(X[trva], (y - anchor)[trva])
        pred = m.predict(X[te])
        imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    p_abs = pred + anchor[te].values
    return {"y": y[te], "pred": pd.Series(p_abs, index=y[te].index), "imp": imp, "te": te, "tr": tr, "va": va,
            "best_iter": getattr(m, "best_iteration_", None)}


def baselines(df, h):
    """単純な基準。persistence: 現在の OT をそのまま。seasonal24: 24 時間前の同時刻の OT。"""
    n = len(df)
    a, b, c = split_bounds(n)
    ot = df["OT"]
    y = ot.shift(-h)
    idx = np.arange(n)
    te = (idx >= b) & (idx < c) & y.notna().values & ot.shift(24).notna().values
    out = {"persistence": ot[te], "seasonal24": ot.shift(24 - h)[te] if h < 24 else ot.shift(48 - h)[te]}
    return y[te], out


def main():
    results = {}
    eda_stats = {}
    for name in ["ETTh1", "ETTh2"]:
        df = load(name)
        a, b, c = split_bounds(len(df))
        eda_stats[name] = {
            "rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
            "OT_mean_by_split": {"train": float(df.OT[:a].mean()), "val": float(df.OT[a:b].mean()), "test": float(df.OT[b:c].mean())},
            "OT_std_by_split": {"train": float(df.OT[:a].std()), "val": float(df.OT[a:b].std()), "test": float(df.OT[b:c].std())},
            "missing": int(df.isna().sum().sum()),
            "corr_with_OT": df.corr()["OT"].drop("OT").round(3).to_dict(),
        }
        res = {}
        # h=0 ナウキャスト（T の負荷から T の OT を推定。T-1 以前の OT のみ使用）
        r_now = run_one(df, 0, True, True, "lgb")
        r_now_nl = run_one(df, 0, False, True, "lgb")
        yb = df["OT"].shift(0)
        ytest = r_now["y"]
        res["nowcast_h0"] = {
            "persistence(T-1)": metrics(ytest, df["OT"].shift(1)[ytest.index]),
            "lgb_OTのみ": metrics(ytest, r_now_nl["pred"][ytest.index]),
            "lgb_OT+負荷": metrics(ytest, r_now["pred"]),
        }
        for h in HORIZONS:
            y, base = baselines(df, h)
            r = {k: metrics(y, v) for k, v in base.items()}
            ridge = run_one(df, h, True, False, "ridge")
            gbm = run_one(df, h, True, False, "lgb")
            gbm_nl = run_one(df, h, False, False, "lgb")
            common = ridge["y"].index.intersection(gbm["y"].index).intersection(gbm_nl["y"].index).intersection(y.index)
            r["ridge_OT+負荷"] = metrics(ridge["y"][common], ridge["pred"][common])
            r["lgb_OTのみ"] = metrics(gbm_nl["y"][common], gbm_nl["pred"][common])
            r["lgb_OT+負荷"] = metrics(gbm["y"][common], gbm["pred"][common])
            r["persistence"] = metrics(y[common], base["persistence"][common])
            r["seasonal24"] = metrics(y[common], base["seasonal24"][common])
            res[f"h{h}"] = r
            if name == "ETTh1":
                gbm["y"].to_frame("y_true").assign(
                    lgb=gbm["pred"], lgb_OTのみ=gbm_nl["pred"], ridge=ridge["pred"],
                    persistence=base["persistence"].reindex(gbm["y"].index)
                ).to_csv(RES / f"ETTh1_test_predictions_h{h}.csv")
                if h == 24:
                    gbm["imp"].sort_values(ascending=False).to_csv(RES / "ETTh1_h24_feature_importance.csv", header=["gain"])
                    r24 = {"y": gbm["y"], "pred": gbm["pred"], "nl": gbm_nl["pred"], "pers": base["persistence"], "imp": gbm["imp"],
                           "best_iter": gbm["best_iter"]}
        results[name] = res
        if name == "ETTh1":
            make_figures(df, r24, results[name])
    results["eda"] = eda_stats
    (RES / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "eda"}, ensure_ascii=False, indent=1)[:6000])


# ---------------------------------------------------------------- 図
def make_figures(df, r24, res):
    a, b, c = split_bounds(len(df))
    # 1. 全期間の OT と分割
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(df.index, df.OT, color=BLUE, lw=0.5)
    for s, e, lab, col in [(0, a, "学習 12か月", "#dbe6f5"), (a, b, "検証 4か月", "#fdeccc"), (b, c, "テスト 4か月", "#d8efe2")]:
        ax.axvspan(df.index[s], df.index[min(e, len(df)) - 1], color=col, alpha=0.8, lw=0)
        ax.text(df.index[s] + (df.index[min(e, len(df)) - 1] - df.index[s]) / 2, df.OT.max() * 0.97, lab, ha="center", fontsize=9)
    ax.set_ylabel("油温 OT [℃]")
    fig.tight_layout()
    fig.savefig(FIGS / "01_ot_overview.png", dpi=170)
    plt.close(fig)
    # 2. 日内パターンと月別
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    g = df.OT.groupby(df.index.hour).mean()
    axs[0].plot(g.index, g.values, color=BLUE, marker="o", ms=3)
    axs[0].set_xlabel("時刻"); axs[0].set_ylabel("平均 OT [℃]"); axs[0].set_title("時刻別の平均油温", fontsize=10)
    m = df.OT.groupby(df.index.to_period("M")).mean()
    axs[1].bar(range(len(m)), m.values, color=BLUE)
    axs[1].set_xticks(range(0, len(m), 3)); axs[1].set_xticklabels([str(p) for p in m.index[::3]], rotation=30, fontsize=7)
    axs[1].set_ylabel("月平均 OT [℃]"); axs[1].set_title("月別の平均油温", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "02_seasonality.png", dpi=170); plt.close(fig)
    # 3. 相関
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, fontsize=8); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGS / "03_corr.png", dpi=170); plt.close(fig)
    # 4. ラグ付き相関（負荷の変化が油温に遅れて効くか）
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lags = range(0, 49, 2)
    for col, color in [("HULL", ORANGE), ("MULL", GREEN), ("LULL", GRAY), ("HUFL", BLUE)]:
        ax.plot(list(lags), [df[col].shift(l).corr(df.OT) for l in lags], label=col, color=color)
    ax.set_xlabel("負荷の遅れ（時間）"); ax.set_ylabel("OT との相関"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "04_lag_corr.png", dpi=170); plt.close(fig)
    # 5. 予測 vs 実測（テスト期間のうち 2 週間）h=24
    fig, ax = plt.subplots(figsize=(10, 3.4))
    y, p, nl, pers = r24["y"], r24["pred"], r24["nl"], r24["pers"]
    s = y.index[len(y) // 2]
    sl = slice(s, s + pd.Timedelta(days=14))
    ax.plot(y[sl].index, y[sl], color="black", lw=1.2, label="実測")
    ax.plot(p[sl].index, p[sl], color=ORANGE, lw=1.2, label="LightGBM（OT＋負荷）")
    ax.plot(pers.reindex(y.index)[sl].index, pers.reindex(y.index)[sl], color=GRAY, lw=0.9, ls="--", label="現在値のまま（基準）")
    ax.set_ylabel("24時間先の OT [℃]"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(); fig.savefig(FIGS / "05_pred_vs_true_h24.png", dpi=170); plt.close(fig)
    # 6. ホライズン別 MAE
    hs = HORIZONS
    labels = {"persistence": ("現在値のまま", GRAY), "seasonal24": ("24h前と同じ", "#b9bec7"),
              "ridge_OT+負荷": ("Ridge", GREEN), "lgb_OTのみ": ("LightGBM（OTのみ）", BLUE), "lgb_OT+負荷": ("LightGBM（OT＋負荷）", ORANGE)}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (lab, col) in labels.items():
        ax.plot(hs, [res[f"h{h}"][k]["MAE"] for h in hs], marker="o", label=lab, color=col)
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([f"{h}h" for h in hs])
    ax.set_xlabel("予測先の時間"); ax.set_ylabel("テスト MAE [℃]"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "06_mae_by_horizon.png", dpi=170); plt.close(fig)
    # 7. 特徴量重要度
    imp = r24["imp"].sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.barh(imp.index, imp.values / imp.values.sum() * 100, color=BLUE)
    ax.set_xlabel("重要度（gain, %）"); ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "07_importance.png", dpi=170); plt.close(fig)
    # 8. 誤差の時間帯別・残差分布
    err = (r24["y"] - r24["pred"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    axs[0].hist(err, bins=60, color=BLUE)
    axs[0].set_xlabel("誤差 実測−予測 [℃]"); axs[0].set_title("24時間先の誤差分布（テスト）", fontsize=10)
    q = err.abs().groupby(err.index.hour).mean()
    axs[1].plot(q.index, q.values, color=ORANGE, marker="o", ms=3)
    axs[1].set_xlabel("予測を出す時刻"); axs[1].set_ylabel("平均絶対誤差 [℃]")
    fig.tight_layout(); fig.savefig(FIGS / "08_error.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
