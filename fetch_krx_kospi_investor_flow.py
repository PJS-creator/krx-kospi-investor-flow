from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

START_DATE = os.getenv("START_DATE", "20250801")
END_DATE = os.getenv("END_DATE", "20260306")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OTP_URL = "https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
DOWNLOAD_URL = "https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
REFERER = "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020301"

HEADERS = {
    "Referer": REFERER,
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

OTP_PAYLOAD = {
    "locale": "ko_KR",
    "inqTpCd": "2",      # 일별추이
    "trdVolVal": "2",    # 거래대금 기준
    "askBid": "3",       # 매도/매수/순매수 모두
    "mktId": "STK",      # KOSPI
    "strtDd": START_DATE,
    "endDd": END_DATE,
    "money": "1",        # 원 단위
    "csvxls_isNo": "false",
    "name": "fileDown",
    "url": "dbms/MDC/STAT/standard/MDCSTAT02202",
}

OUTFILE = OUTPUT_DIR / f"kospi_investor_flow_daily_{START_DATE}_{END_DATE}_from_krx.csv"
RAWFILE = OUTPUT_DIR / f"krx_raw_investor_flow_{START_DATE}_{END_DATE}.csv"
LOGFILE = OUTPUT_DIR / f"run_log_{START_DATE}_{END_DATE}.txt"


def log(msg: str) -> None:
    print(msg)
    with LOGFILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _normalize_col(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def _pick(df: pd.DataFrame, include: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for col in df.columns:
        norm = _normalize_col(col)
        if all(x in norm for x in include) and not any(x in norm for x in exclude):
            return col
    return None


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick(df, ["일자"]) or _pick(df, ["날짜"])
    if date_col is None:
        raise ValueError(f"날짜 컬럼을 찾지 못했습니다. 실제 컬럼: {list(df.columns)}")

    out = pd.DataFrame()
    out["date_kr"] = pd.to_datetime(df[date_col].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")

    patterns = {
        "foreign_sell_amt": [["외국인", "매도"], ["외국인합계", "매도"]],
        "foreign_buy_amt": [["외국인", "매수"], ["외국인합계", "매수"]],
        "foreign_net_amt": [["외국인", "순매수"], ["외국인합계", "순매수"]],
        "individual_sell_amt": [["개인", "매도"]],
        "individual_buy_amt": [["개인", "매수"]],
        "individual_net_amt": [["개인", "순매수"]],
        "institution_sell_amt": [["기관", "매도"], ["기관합계", "매도"]],
        "institution_buy_amt": [["기관", "매수"], ["기관합계", "매수"]],
        "institution_net_amt": [["기관", "순매수"], ["기관합계", "순매수"]],
    }

    found = {}
    for out_col, pattern_sets in patterns.items():
        chosen = None
        for pats in pattern_sets:
            chosen = _pick(df, pats)
            if chosen:
                break
        found[out_col] = chosen
        if chosen:
            out[out_col] = _to_num(df[chosen])
        else:
            out[out_col] = pd.NA

    # 이 스크립트는 투자자별 거래실적만 우선 추출한다.
    # total_kospi_trading_value는 별도 KRX 다운로드 단계가 필요하므로 비워둔다.
    out["total_kospi_trading_value"] = pd.NA

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
    log(f"START_DATE={START_DATE}")
    log(f"END_DATE={END_DATE}")
    log("KRX 투자자별 거래실적 다운로드 시작")

    session = requests.Session()
    session.headers.update(HEADERS)

    # 쿠키 세팅용
    pre = session.get(REFERER, timeout=30)
    pre.raise_for_status()
    log(f"Referer page status: {pre.status_code}")

    payload = dict(OTP_PAYLOAD)
    payload["strtDd"] = START_DATE
    payload["endDd"] = END_DATE

    otp = session.post(OTP_URL, data=payload, timeout=30)
    otp.raise_for_status()
    code = otp.text.strip()
    if not code:
        raise RuntimeError("KRX OTP 코드가 비어 있습니다.")
    log(f"OTP length: {len(code)}")

    resp = session.post(DOWNLOAD_URL, data={"code": code}, timeout=60)
    resp.raise_for_status()
    RAWFILE.write_bytes(resp.content)
    log(f"Raw bytes saved: {RAWFILE}")

    try:
        df = pd.read_csv(BytesIO(resp.content), encoding="euc-kr")
    except Exception as e:
        log(f"CSV 파싱 실패: {e}")
        raise

    log(f"Raw shape: {df.shape}")
    log(f"Raw columns: {list(df.columns)}")

    out = _extract(df)
    out.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
    log(f"Saved processed file: {OUTFILE}")
    log(f"Processed rows: {len(out)}")
    log("완료")


if __name__ == "__main__":
    main()
