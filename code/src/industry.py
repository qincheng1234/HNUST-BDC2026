"""Industry classification — downloaded once during training, reused for prediction.

The resulting file lives in the model output directory so it survives the
Docker mounts (data/, output/, temp/ are overwritten at verification time).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_industry_map(industry_path):
    """Return {stock_id: industry_name} from a CSV, or {} on failure."""
    path = Path(industry_path)
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path, dtype={"stock_id": str})
        if "industry" not in df.columns:
            return {}
        return dict(zip(df["stock_id"], df["industry"]))
    except Exception:
        return {}


def ensure_industry_data(stock_ids, model_output_dir, data_dir="./data"):
    """Make sure sw_industry.csv exists in *model_output_dir*.

    If the file is already present we do nothing.  Otherwise we try to
    download from Tushare.  Falls back to a coarse board-level grouping
    when Tushare is unavailable or the token is missing.
    """
    model_dir = Path(model_output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "sw_industry.csv"
    if target.is_file() and target.stat().st_size > 0:
        existing = load_industry_map(target)
        if len(existing) >= 150:
            print(f"  行业数据已存在: {target} ({len(existing)} stocks)")
            return target

    print("  行业数据缺失，尝试从 Tushare 下载…")

    industry_map = _download_via_tushare(stock_ids)
    if len(industry_map) < 150:
        industry_map = _board_fallback(stock_ids)

    # Fill missing
    for sid in stock_ids:
        industry_map.setdefault(sid, "未分类")

    output = pd.DataFrame(
        {"stock_id": k, "industry": v}
        for k, v in sorted(industry_map.items())
    )
    output.to_csv(target, index=False, encoding="utf-8-sig")

    # Also save to data/ for local convenience
    data_target = Path(data_dir) / "sw_industry.csv"
    try:
        data_target.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(data_target, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    counts = output["industry"].value_counts()
    print(f"  行业数据已保存: {target}")
    print(f"  覆盖 {len(industry_map)} stocks, {len(counts)} sectors")
    return target


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _resolve_tushare_token():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    mcp_url = os.environ.get("TUSHARE_MCP_URL", "").strip()
    if mcp_url:
        token_values = parse_qs(urlparse(mcp_url).query).get("token", [])
        if token_values and token_values[0].strip():
            return token_values[0].strip()
    token_file = Path(os.environ.get("TUSHARE_TOKEN_FILE", "./temp/tushare_token.txt"))
    try:
        return token_file.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _get_pro():
    try:
        import tushare as ts
    except ImportError:
        return None
    token = _resolve_tushare_token()
    if not token:
        return None
    return ts.pro_api(token)


def _retry(func, desc, retries=4, wait=3):
    for attempt in range(1, retries + 1):
        try:
            result = func()
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                return result
        except Exception as e:
            if attempt >= retries:
                print(f"  {desc} 最终失败: {e}")
                return None
            print(f"  {desc} 失败，{wait}s 后重试 ({attempt}/{retries}): {e}")
            time.sleep(wait)
    return None


def _download_via_tushare(stock_ids):
    """Try Shenwan 2021 L1, then CSRC industry from stock_basic."""
    pro = _get_pro()
    if pro is None:
        print("  Tushare 不可用，跳过行业下载")
        return {}

    stock_set = set(stock_ids)
    industry_map: dict[str, str] = {}

    # 1) Shenwan 2021 L1
    print("  尝试申万2021一级行业分类…")
    sw = _retry(
        lambda: pro.index_classify(level="L1", src="SW2021"),
        desc="申万行业分类",
    )
    if sw is not None and not sw.empty:
        sw = sw.rename(columns={"index_code": "stock_id"})
        sw["stock_id"] = sw["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False)
        sw = sw.dropna(subset=["stock_id"])
        sw_in = sw[sw["stock_id"].isin(stock_set)]
        if len(sw_in) >= 150:
            industry_map = dict(zip(sw_in["stock_id"], sw_in["industry_name"]))
            print(f"    申万行业分类覆盖 {len(industry_map)}/{len(stock_set)}")

    # 2) CSRC fallback
    if len(industry_map) < 150:
        print("  尝试证监会行业分类…")
        basic = _retry(
            lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry"),
            desc="股票基础信息",
        )
        if basic is not None and not basic.empty:
            basic["stock_id"] = basic["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False)
            basic = basic.dropna(subset=["stock_id", "industry"])
            basic["industry"] = basic["industry"].str.strip()
            basic = basic[basic["industry"] != ""]
            csrc = basic[basic["stock_id"].isin(stock_set)]
            if len(csrc) >= 150:
                industry_map = dict(zip(csrc["stock_id"], csrc["industry"]))
                print(f"    证监会行业分类覆盖 {len(industry_map)}/{len(stock_set)}")

    return industry_map


def _board_fallback(stock_ids):
    """Coarse exchange/board grouping when no external data is available."""
    mapping = {}
    for sid in stock_ids:
        if sid.startswith("688"):
            mapping[sid] = "科创板"
        elif sid.startswith("300") or sid.startswith("301"):
            mapping[sid] = "创业板"
        elif sid.startswith("000") or sid.startswith("001") or sid.startswith("002") or sid.startswith("003"):
            mapping[sid] = "深市主板"
        elif sid.startswith("600") or sid.startswith("601") or sid.startswith("603") or sid.startswith("605"):
            mapping[sid] = "沪市主板"
        else:
            mapping[sid] = "未分类"
    return mapping
