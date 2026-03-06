from __future__ import annotations

import csv
import os
import re
from io import StringIO
from pathlib import Path
from typing import Iterable, List

import pandas as pd

RAW_INPUT_PATH = os.getenv("RAW_INPUT_PATH", "krx_investor_flow_raw.csv")
START_DATE = os.getenv("START_DATE", "20250801")
END_DATE = os.getenv("END_DATE", "20260306")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTFILE = OUTPUT_DIR / f"kospi_investor_flow_daily_{START_DATE}_{END_DATE}_from_uploaded_krx.csv"
LOGFILE = OUTPUT_DIR / f"process_log_{START_DATE}_{END_DATE}.txt"


def log(msg: str) -> None:
    print(msg)
    with LOGFILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _normalize_text(x: object) -> str:
    s = str(x) if x is not None else ""
    s = s.replace("\ufeff", "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)
    return s.strip()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()

    # 명시적인 차단/에러 응답 감지
    raw_head = raw[:5000].decode("utf-8", errors="ignore")
    if "LOGOUT" in raw_head.upper():
        raise RuntimeError(
            "업로드한 파일 내용에 LOGOUT이 들어 있습니다. KRX 로그인 없이 받은 응답일 가능성이 큽니다. "
            "브라우저에서 KRX에 로그인한 뒤, 화면에서 직접 CSV를 다시 내려받아 업로드해 주세요."
        )
    if "<html" in raw_head.lower() or "서비스 에러" in raw_head:
        raise RuntimeError(
            "업로드한 파일이 CSV가 아니라 HTML/에러 페이지입니다. KRX 화면에서 CSV로 다시 다운로드해 주세요."
        )

    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1")


def _guess_delimiter(sample: str) -> str:
    comma = sample.count(",")
    tab = sample.count("\t")
    semicolon = sample.count(";")
    if tab > comma and tab > semicolon:
        return "\t"
    if semicolon > comma and semicolon > tab:
        return ";"
    return ","


def _ffill(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    last = ""
    for v in values:
        norm = _normalize_text(v)
        if norm:
            last = norm
            out.append(norm)
        else:
            out.append(last)
    return out


def _is_subheader_row(row: list[str]) -> bool:
    labels = {"매도", "매수", "순매수", "매도금액", "매수금액", "순매수금액"}
    normed = [_normalize_text(x) for x in row]
    hit = sum(1 for x in normed if x in labels)
    return hit >= 3


def _find_header_idx(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows[:20]):
        normed = [_normalize_text(x) for x in row]
        joined = "|".join(normed)
        if ("일자" in joined or "날짜" in joined) and any(
            key in joined for key in ["외국인", "개인", "기관", "전체"]
        ):
            return i
    raise RuntimeError(
        "헤더 행을 찾지 못했습니다. 업로드한 파일이 KRX 투자자별 거래실적 CSV인지 확인해 주세요."
    )


def _build_dataframe_from_text(text: str) -> pd.DataFrame:
    delim = _guess_delimiter(text[:2000])
    rows = list(csv.reader(StringIO(text), delimiter=delim))
    rows = [row for row in rows if any(_normalize_text(c) for c in row)]
    if not rows:
        raise RuntimeError("파일이 비어 있습니다.")

    header_idx = _find_header_idx(rows)
    h1 = rows[header_idx]
    h2 = rows[header_idx + 1] if header_idx + 1 < len(rows) else []

    # 길이 맞추기
    ncols = max(len(r) for r in rows[header_idx : min(len(rows), header_idx + 5)])
    def pad(r: list[str]) -> list[str]:
        return r + [""] * (ncols - len(r))

    h1 = pad(h1)
    h2 = pad(h2)

    use_two_header = _is_subheader_row(h2)

    if use_two_header:
        top = _ffill(h1)
        sub = [_normalize_text(x) for x in h2]
        columns = []
        for a, b in zip(top, sub):
            if b and a and a != b:
                columns.append(f"{a}_{b}")
            elif a:
                columns.append(a)
            else:
                columns.append(b)
        data_start = header_idx + 2
    else:
        columns = [_normalize_text(x) for x in h1]
        data_start = header_idx + 1

    data_rows: list[list[str]] = []
    for row in rows[data_start:]:
        row = pad(row)
        # 완전 빈 줄은 제외
        if not any(_normalize_text(c) for c in row):
            continue
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=columns)

    # 의미 없는 칼럼 제거
    keep_cols = []
    for c in df.columns:
        norm = _normalize_text(c)
        if not norm:
            continue
        if norm.startswith("Unnamed"):
            continue
        keep_cols.append(c)
    df = df[keep_cols]
    return df


def _pick(df: pd.DataFrame, include: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for col in df.columns:
        norm = _normalize_text(col)
        if all(x in norm for x in include) and not any(x in norm for x in exclude):
            return col
    return None


def _to_num(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("--", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_date(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
        .str.strip()
    )
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick(df, ["일자"]) or _pick(df, ["날짜"])
    if date_col is None:
        raise RuntimeError(f"날짜 컬럼을 찾지 못했습니다. 실제 컬럼: {list(df.columns)}")

    out = pd.DataFrame()
    out["date_kr"] = _to_date(df[date_col])

    patterns = {
        "foreign_sell_amt": [["외국인합계", "매도"], ["외국인", "매도"]],
        "foreign_buy_amt": [["외국인합계", "매수"], ["외국인", "매수"]],
        "foreign_net_amt": [["외국인합계", "순매수"], ["외국인", "순매수"]],
        "individual_sell_amt": [["개인", "매도"]],
        "individual_buy_amt": [["개인", "매수"]],
        "individual_net_amt": [["개인", "순매수"]],
        "institution_sell_amt": [["기관합계", "매도"], ["기관", "매도"]],
        "institution_buy_amt": [["기관합계", "매수"], ["기관", "매수"]],
        "institution_net_amt": [["기관합계", "순매수"], ["기관", "순매수"]],
        "_total_buy": [["전체", "매수"], ["합계", "매수"]],
        "_total_sell": [["전체", "매도"], ["합계", "매도"]],
    }

    found: dict[str, str | None] = {}
    for out_col, pattern_sets in patterns.items():
        chosen = None
        for pats in pattern_sets:
            chosen = _pick(df, pats)
            if chosen:
                break
        found[out_col] = chosen
        if out_col.startswith("_"):
            continue
        out[out_col] = _to_num(df[chosen]) if chosen else pd.NA

    total_col = found.get("_total_buy") or found.get("_total_sell")
    out["total_kospi_trading_value"] = _to_num(df[total_col]) if total_col else pd.NA

    ordered = [
        "date_kr",
        "foreign_buy_amt",
        "foreign_sell_amt",
        "foreign_net_amt",
        "total_kospi_trading_value",
        "individual_buy_amt",
        "individual_sell_amt",
        "individual_net_amt",
        "institution_buy_amt",
        "institution_sell_amt",
        "institution_net_amt",
    ]
    out = out[ordered].dropna(subset=["date_kr"]).sort_values("date_kr").reset_index(drop=True)

    log("[컬럼 매핑 결과]")
    for k, v in found.items():
        log(f"- {k}: {v}")

    return out


def main() -> None:
    if LOGFILE.exists():
        LOGFILE.unlink()

    raw_path = Path(RAW_INPUT_PATH)
    log(f"RAW_INPUT_PATH={raw_path}")
    log(f"START_DATE={START_DATE}")
    log(f"END_DATE={END_DATE}")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾지 못했습니다: {raw_path}. 먼저 KRX에서 직접 내려받은 CSV를 저장소에 업로드해 주세요."
        )

    text = _read_text(raw_path)
    df_raw = _build_dataframe_from_text(text)
    log(f"Raw shape: {df_raw.shape}")
    log(f"Raw columns: {list(df_raw.columns)}")

    out = _extract(df_raw)

    start_fmt = pd.to_datetime(START_DATE, format="%Y%m%d").strftime("%Y-%m-%d")
    end_fmt = pd.to_datetime(END_DATE, format="%Y%m%d").strftime("%Y-%m-%d")
    out = out[(out["date_kr"] >= start_fmt) & (out["date_kr"] <= end_fmt)].copy()

    if out.empty:
        raise RuntimeError(
            f"가공 후 남은 행이 없습니다. 업로드한 파일 날짜 구간과 입력 날짜({START_DATE}~{END_DATE})를 확인해 주세요."
        )

    out.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
    log(f"Saved processed file: {OUTFILE}")
    log(f"Processed rows: {len(out)}")
    log(f"Date range in output: {out['date_kr'].min()} ~ {out['date_kr'].max()}")
    log("완료")


if __name__ == "__main__":
    main()
