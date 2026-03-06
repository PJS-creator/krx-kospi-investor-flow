
from __future__ import annotations

import csv
import os
import re
from io import StringIO
from pathlib import Path
from typing import Iterable

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

    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1")


def _guess_delimiter(sample: str) -> str:
    counts = {",": sample.count(","), "\t": sample.count("\t"), ";": sample.count(";")}
    return max(counts, key=counts.get) if any(counts.values()) else ","


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
    return sum(1 for x in normed if x in labels) >= 2


def _find_header_idx(rows: list[list[str]]) -> int:
    for i, row in enumerate(rows[:20]):
        normed = [_normalize_text(x) for x in row]
        joined = "|".join(normed)
        if ("일자" in joined or "날짜" in joined) and any(
            key in joined for key in ["외국인", "개인", "기관", "전체", "기타법인"]
        ):
            return i
    raise RuntimeError("헤더 행을 찾지 못했습니다. 업로드한 파일이 KRX 투자자별 거래실적 CSV인지 확인해 주세요.")


def _build_dataframe_from_text(text: str) -> pd.DataFrame:
    delim = _guess_delimiter(text[:2000])
    rows = list(csv.reader(StringIO(text), delimiter=delim))
    rows = [row for row in rows if any(_normalize_text(c) for c in row)]
    if not rows:
        raise RuntimeError("파일이 비어 있습니다.")

    header_idx = _find_header_idx(rows)
    h1 = rows[header_idx]
    h2 = rows[header_idx + 1] if header_idx + 1 < len(rows) else []

    ncols = max(len(r) for r in rows[header_idx:min(len(rows), header_idx + 5)])

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
            a = _normalize_text(a)
            b = _normalize_text(b)
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
        if not any(_normalize_text(c) for c in row):
            continue
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=columns)
    keep_cols = [c for c in df.columns if _normalize_text(c) and not _normalize_text(c).startswith("Unnamed")]
    return df[keep_cols]


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
        .str.replace('"', "", regex=False)
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


def _base_output(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date_kr"] = _to_date(df[date_col])
    for c in [
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
    ]:
        out[c] = pd.NA
    return out


def _extract_multi_metric(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, dict[str, str | None]]:
    out = _base_output(df, date_col)

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
        if chosen:
            out[out_col] = _to_num(df[chosen])

    total_col = found.get("_total_buy") or found.get("_total_sell")
    if total_col:
        out["total_kospi_trading_value"] = _to_num(df[total_col])

    return out, found


def _extract_net_only(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, dict[str, str | None]]:
    out = _base_output(df, date_col)

    found = {
        "foreign_sell_amt": None,
        "foreign_buy_amt": None,
        "foreign_net_amt": _pick(df, ["외국인합계"]) or _pick(df, ["외국인"]),
        "individual_sell_amt": None,
        "individual_buy_amt": None,
        "individual_net_amt": _pick(df, ["개인"]),
        "institution_sell_amt": None,
        "institution_buy_amt": None,
        "institution_net_amt": _pick(df, ["기관합계"]) or _pick(df, ["기관"]),
        "_total_buy": None,
        "_total_sell": None,
    }

    for out_col in ["foreign_net_amt", "individual_net_amt", "institution_net_amt"]:
        src = found[out_col]
        if src:
            out[out_col] = _to_num(df[src])

    total_col = _pick(df, ["전체"])
    if total_col is not None:
        total_numeric = _to_num(df[total_col])
        # 순매수 합계(이론상 0)에 해당하는 컬럼일 가능성이 높으므로,
        # 거의 전부 0이면 거래대금으로 오인하지 않도록 비워둔다.
        if (total_numeric.fillna(0).abs() > 0).sum() > 0:
            # 값이 0이 아닌 경우에도 '전체'가 순매수합계일 가능성이 높아 기본적으로 사용하지 않는다.
            log("경고: '전체' 컬럼이 있으나 이 파일 형식에서는 보통 순매수 합계(대체로 0)이므로 total_kospi_trading_value로 사용하지 않았습니다.")
        else:
            log("참고: 업로드한 파일의 '전체' 컬럼은 전부 0으로, 총 거래대금 컬럼이 아닙니다.")
    log("참고: 업로드한 CSV는 투자자별 '순매수만' 있는 형식으로 보입니다. 매수/매도/총거래대금은 이 파일만으로 복원할 수 없습니다.")

    return out, found


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick(df, ["일자"]) or _pick(df, ["날짜"])
    if date_col is None:
        raise RuntimeError(f"날짜 컬럼을 찾지 못했습니다. 실제 컬럼: {list(df.columns)}")

    has_metric_headers = any(any(token in _normalize_text(c) for token in ["매도", "매수", "순매수"]) for c in df.columns)
    net_only_shape = (
        (_pick(df, ["외국인합계"]) or _pick(df, ["외국인"])) is not None
        and (_pick(df, ["개인"]) is not None)
        and ((_pick(df, ["기관합계"]) or _pick(df, ["기관"])) is not None)
        and not has_metric_headers
    )

    if has_metric_headers:
        mode = "multi_metric"
        out, found = _extract_multi_metric(df, date_col)
    elif net_only_shape:
        mode = "net_only"
        out, found = _extract_net_only(df, date_col)
    else:
        raise RuntimeError(
            "알 수 없는 CSV 형식입니다. 현재 스크립트는 (1) 매수/매도/순매수 3종 컬럼형, "
            "(2) 투자자별 순매수만 있는 형식을 지원합니다. "
            f"실제 컬럼: {list(df.columns)}"
        )

    out = out.dropna(subset=["date_kr"]).sort_values("date_kr").reset_index(drop=True)

    log(f"[감지된 형식] {mode}")
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
