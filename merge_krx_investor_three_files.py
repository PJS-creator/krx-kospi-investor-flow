#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd


BUY_PATH = os.getenv("BUY_PATH", "krx_kospi_investor_buy_amt.csv")
SELL_PATH = os.getenv("SELL_PATH", "krx_kospi_investor_sell_amt.csv")
NET_PATH = os.getenv("NET_PATH", "krx_kospi_investor_net_amt.csv")
OUT_DIR = Path(os.getenv("OUT_DIR", "output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("euc-kr", "cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read {path}: {last_err}")


def get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {re.sub(r"\s+", "", c): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", "", cand)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    date_col = get_col(df, ["일자", "날짜", "date"])
    foreign_col = get_col(df, ["외국인 합계", "외국인합계"])
    individual_col = get_col(df, ["개인"])
    institution_col = get_col(df, ["기관 합계", "기관합계"])
    total_col = get_col(df, ["전체"])

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], format="%Y/%m/%d", errors="coerce")
    if out[date_col].isna().any():
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    out["date_kr"] = out[date_col].dt.strftime("%Y-%m-%d")
    out = out.rename(
        columns={
            foreign_col: "foreign",
            individual_col: "individual",
            institution_col: "institution",
            total_col: "total",
        }
    )
    return out[["date_kr", "foreign", "individual", "institution", "total"]]


def main() -> None:
    print(f"BUY_PATH={BUY_PATH}")
    print(f"SELL_PATH={SELL_PATH}")
    print(f"NET_PATH={NET_PATH}")

    buy = prep(read_csv_auto(BUY_PATH))
    sell = prep(read_csv_auto(SELL_PATH))
    net = prep(read_csv_auto(NET_PATH))

    if not buy["date_kr"].equals(sell["date_kr"]) or not buy["date_kr"].equals(net["date_kr"]):
        raise ValueError("The three files do not have the same date rows in the same order.")

    out = pd.DataFrame(
        {
            "date_kr": buy["date_kr"],
            "foreign_buy_amt": buy["foreign"],
            "foreign_sell_amt": sell["foreign"],
            "foreign_net_amt": net["foreign"],
            "total_kospi_trading_value": buy["total"],
            "individual_buy_amt": buy["individual"],
            "individual_sell_amt": sell["individual"],
            "individual_net_amt": net["individual"],
            "institution_buy_amt": buy["institution"],
            "institution_sell_amt": sell["institution"],
            "institution_net_amt": net["institution"],
        }
    )

    for col in out.columns[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # basic QC
    qc = []
    qc.append(f"Rows: {len(out)}")
    qc.append(f"Date range: {out['date_kr'].min()} ~ {out['date_kr'].max()}")
    qc.append(f"All total_kospi_trading_value same between buy/sell: {(buy['total'] == sell['total']).all()}")
    qc.append(f"All total in net file equal zero: {(net['total'].fillna(0) == 0).all()}")

    for label, bcol, scol, ncol in [
        ("foreign", "foreign", "foreign", "foreign"),
        ("individual", "individual", "individual", "individual"),
        ("institution", "institution", "institution", "institution"),
    ]:
        diff = buy[bcol] - sell[scol] - net[ncol]
        qc.append(
            f"{label} | exact (buy-sell==net) rows: {(diff.fillna(0) == 0).sum()} / {len(diff)}, max_abs_diff: {diff.abs().max()}"
        )

    out_file = OUT_DIR / "kospi_investor_flow_daily_from_three_krx_files.csv"
    log_file = OUT_DIR / "merge_qc_log.txt"

    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    log_file.write_text("\n".join(qc), encoding="utf-8")

    print(f"Saved: {out_file}")
    print(f"Saved: {log_file}")


if __name__ == "__main__":
    main()
