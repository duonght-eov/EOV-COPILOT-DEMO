import httpx
from langchain_core.tools import Tool
from app.core.config import get_settings


def _parse_forecast_records(data: dict) -> list:
    """Extract records từ cấu trúc trả về thực tế"""
    try:
        # Check cấu trúc data.content[0].data (Hệ thống hiện hành)
        content_list = data.get("data", {}).get("content", [])
        if isinstance(content_list, list) and content_list:
            records = content_list[0].get("data", [])
            if isinstance(records, list) and records:
                return records
                
        # Fallback 1: cấu trúc theo API Guide (data.records)
        records = data.get("data", {}).get("records", [])
        if isinstance(records, list) and records:
            return records
            
        # Fallback 2... (tìm tự động)
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    except Exception:
        pass
    return []


def _format_forecast_table(records: list, dma_id: str, horizon: str, limit: int = 12) -> str:
    """Format records thành Markdown table."""
    if not records:
        return f"Không có dữ liệu dự báo cho DMA {dma_id}."
        
    # Giới hạn số tháng để bảng không quá dài, luôn lấy dữ liệu mới nhất (nằm ở cuối list do đã sort)
    sorted_rows = sorted(records, key=lambda r: str(r.get("year_month", "")))
    recent = sorted_rows[-limit:]
    
    title_prefix = "Dự báo" if horizon in ["ngắn hạn", "dài hạn"] else "Lịch sử tiêu thụ"
    header = f"**{title_prefix} DMA {dma_id.upper()}** ({len(recent)} tháng):\n\n"
    
    # Do khác nhau giữa thực tế và Guide, ta build dynamic table
    first_row = recent[0]
    has_safe_raw = "prediction_safe" in first_row
    
    if has_safe_raw:
        header += "| Tháng | Dự báo an toàn (m³) | Dự báo thô (m³) | Buffer (m³) | Thiếu hụt |\n"
        header += "|-------|---------------------|-----------------|------------|----------|\n"
        rows = []
        for r in recent:
            ym = str(r.get("year_month", "?"))[:7]
            safe  = f"{r['prediction_safe']:,.0f}"  if r.get("prediction_safe")  is not None else "-"
            raw   = f"{r['prediction_raw']:,.0f}"   if r.get("prediction_raw")   is not None else "-"
            buf   = f"{r['buffer_volume']:,.0f}"    if r.get("buffer_volume")    is not None else "-"
            short = "⚠️ Có" if r.get("is_shortage") else "✅ Không"
            rows.append(f"| {ym} | {safe} | {raw} | {buf} | {short} |")
    else:
        # Dữ liệu dạng ngang cho Lịch sử / Dự báo cơ bản
        if horizon == "lịch sử":
            title_col = "Tiêu thụ thực tế (m³)"
            # Render bảng ngang
            header += "| Tháng | " + " | ".join(str(r.get("year_month", "?"))[:7] for r in recent) + " |\n"
            header += "|-------|" + "|".join(["---"] * len(recent)) + "|\n"
            
            row_vals = []
            for r in recent:
                val = r.get("actual") if r.get("actual") is not None else r.get("predicted_demand")
                val_str = f"{val:,.0f}" if val is not None else "-"
                row_vals.append(val_str)
            
            return header + f"| {title_col} | " + " | ".join(row_vals) + " |\n"
        else:
            header += "| Tháng | Dự báo tiêu thụ (m³) |\n"
            header += "|-------|-----------------------------|\n"
            rows = []
            for r in recent:
                ym = str(r.get("year_month", "?"))[:7]
                val = r.get("predicted_demand")
                val_str = f"{val:,.0f}" if val is not None else "-"
                rows.append(f"| {ym} | {val_str} |")
            return header + "\n".join(rows)


async def _get_short_term_forecast(dma_id: str, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas/{dma_id}/forecasts?horizon=short-term"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            records = _parse_forecast_records(resp.json())
            # Ngắn hạn (Short-Term T+1) -> Slice đúng 1 tháng dự báo tương lai
            return _format_forecast_table(records, dma_id, "ngắn hạn", limit=1)
    except Exception as e:
        return f"[Lỗi lấy dự báo ngắn hạn DMA {dma_id}: {e}]"


async def _get_long_term_forecast(dma_id: str, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas/{dma_id}/forecasts?horizon=long-term"
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            records = _parse_forecast_records(resp.json())
            # Dài hạn (Long-Term T+1 -> T+3) -> Slice đúng 3 tháng dự báo/quý
            return _format_forecast_table(records, dma_id, "dài hạn", limit=3)
    except Exception as e:
        return f"[Lỗi lấy dự báo dài hạn DMA {dma_id}: {e}]"


async def _get_history(dma_id: str, months: int = 24, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/history?dma={dma_id}&months={months}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            records = _parse_forecast_records(resp.json())
            if records:
                # Lịch sử theo số tháng do user yêu cầu
                return _format_forecast_table(records, dma_id, "lịch sử", limit=months)
            return f"Không có dữ liệu lịch sử cho DMA {dma_id}."
    except Exception as e:
        return f"[Lỗi lấy lịch sử DMA {dma_id}: {e}]"





async def _get_dma_list(base_url: str | None = None, api_key: str | None = None) -> str:
    """Lấy danh sách toàn bộ khu vực DMA trong hệ thống."""
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas?limit=1000"
    key = api_key or settings.predict_api_key
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"X-API-Key": key})
            resp.raise_for_status()
            data = resp.json()

            def _find_list(obj, depth=0):
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict) and depth < 3:
                    for v in obj.values():
                        result = _find_list(v, depth + 1)
                        if result is not None:
                            return result
                return None

            items = _find_list(data)
            if isinstance(items, list) and items:
                ids = []
                for d in items:
                    dma_id = d.get("dma_id") or d.get("id", "?") if isinstance(d, dict) else str(d)
                    ids.append(str(dma_id))

                total = len(ids)
                cols = 8
                header_row = "| " + " | ".join(str(i + 1) for i in range(cols)) + " |"
                sep_row    = "| " + " | ".join(["---"] * cols) + " |"
                rows = []
                for i in range(0, total, cols):
                    chunk = ids[i:i + cols]
                    chunk += [""] * (cols - len(chunk))
                    rows.append("| " + " | ".join(f"`{c}`" if c else "" for c in chunk) + " |")

                table = header_row + "\n" + sep_row + "\n" + "\n".join(rows)
                return f"**{total} mã DMA trong hệ thống:**\n\n" + table

            return "Không thể trích xuất danh sách DMA. Raw: " + str(data)[:300]
    except Exception as e:
        return f"[Lỗi lấy danh sách DMA: {e}]"


def make_predict_tools(base_url: str | None = None, api_key: str | None = None) -> list[Tool]:
    async def run_short_term(dma_id: str) -> str:
        return await _get_short_term_forecast(dma_id.strip(), base_url, api_key)

    async def run_long_term(dma_id: str) -> str:
        return await _get_long_term_forecast(dma_id.strip(), base_url, api_key)

    async def run_history(dma_id: str) -> str:
        # Nếu LLM call dạng Function, ta mặc định lấy 24 (nếu user gửi số tháng thì Router xử lý)
        return await _get_history(dma_id.strip(), base_url=base_url, api_key=api_key)

    async def run_dma_list(_: str) -> str:
        return await _get_dma_list(base_url, api_key)

    return [
        Tool(name="get_short_term_forecast",
             description="Lấy dự báo ngắn hạn (T+1) của một DMA. Input: mã DMA (vd: '78-Vin', '01-BT').",
             coroutine=run_short_term, func=lambda q: None),
        Tool(name="get_long_term_forecast",
             description="Lấy dự báo dài hạn T+1→T+3 (1 quý) của một DMA. Input: mã DMA.",
             coroutine=run_long_term, func=lambda q: None),
        Tool(name="get_history",
             description="Lấy lịch sử tiêu thụ 24 tháng qua của một DMA. Input: mã DMA.",
             coroutine=run_history, func=lambda q: None),
        Tool(name="get_dma_list",
             description="Lấy danh sách toàn bộ mã DMA. Input: chuỗi rỗng.",
             coroutine=run_dma_list, func=lambda q: None),
    ]
