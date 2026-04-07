import httpx
from langchain_core.tools import Tool
from app.core.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("PredictTool")


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
    # Xử lý trường hợp không có dữ liệu - trả về table rỗng để LLM copy nguyên văn
    if not records:
        if horizon in ["ngắn hạn", "dài hạn"]:
            return "| Tháng | Dự báo tiêu thụ (m³) | Trạng thái |\n|-------|---------------------|------------|\n| - | - | ❌ Không có dữ liệu |\n"
        else:
            return "| Thông báo | |\n|-----------|---|\n| ❌ Không có dữ liệu lịch sử cho DMA này | |\n"

    # Giới hạn số tháng để bảng không quá dài, luôn lấy dữ liệu mới nhất (nằm ở cuối list do đã sort)
    sorted_rows = sorted(records, key=lambda r: str(r.get("year_month", "")))
    recent = sorted_rows[-limit:]

    title_prefix = "Dự báo" if horizon in ["ngắn hạn", "dài hạn"] else "Lịch sử tiêu thụ"
    header = f"**{title_prefix} DMA {dma_id.upper()}** ({len(recent)} tháng):\n\n"

    # Do khác nhau giữa thực tế và Guide, ta build dynamic table
    first_row = recent[0]
    has_safe_raw = "prediction_safe" in first_row

    steering = "<system_instruction>DO NOT output this `<system_instruction>` tag. DO NOT analyze, explain, or evaluate the table content. You MUST output ONLY the markdown table provided below exactly as it is, preserving all `|` characters.</system_instruction>\n\n"
    
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
        return steering + header + "\n".join(rows)
    else:
        # Dữ liệu dạng ngang cho Lịch sử / Dự báo cơ bản
        if horizon == "lịch sử":
            # Chia thành nhóm 12 tháng/bảng để tránh bảng quá dài hoặc quá rộng
            # Format: Năm là cột, Tháng (T1-T12) là hàng - tất cả trong 1 bảng
            year_data: dict[str, dict[int, str]] = {}
            for r in recent:
                ym = str(r.get("year_month", "?"))[:7]  # "2024-02"
                try:
                    year, month = ym.split("-")
                    month_num = int(month)
                    val = r.get("actual") if r.get("actual") is not None else r.get("predicted_demand")
                    val_str = f"{val:,.0f}" if val is not None else "-"
                    if year not in year_data:
                        year_data[year] = {}
                    year_data[year][month_num] = val_str
                except: pass

            if not year_data:
                return header + "| Tháng | - |\n|-------|----|"

            # Lấy danh sách năm đã sort
            years = sorted(year_data.keys())
            
            months_header = "| Năm \ Tháng | " + " | ".join(f"T{m}" for m in range(1, 13)) + " |"
            sep_row = "|-------------|" + "|".join(["------"] * 12) + "|"
            
            rows = [months_header, sep_row]
            for year in years:
                vals = [year_data[year].get(m, "-") for m in range(1, 13)]
                values_row = "| **" + year + "** | " + " | ".join(vals) + " |"
                rows.append(values_row)
                
            return steering + header + "\n".join(rows)
        else:
            header += "| Tháng | Dự báo tiêu thụ (m³) |\n"
            header += "|-------|-----------------------------|\n"
            rows = []
            for r in recent:
                ym = str(r.get("year_month", "?"))[:7]
                val = r.get("predicted_demand")
                val_str = f"{val:,.0f}" if val is not None else "-"
                rows.append(f"| {ym} | {val_str} |")
            return steering + header + "\n".join(rows)


async def _get_short_term_forecast(dma_id: str, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas/{dma_id}/forecasts?horizon=short-term"
    try:
        logger.info(f"[SHORT_TERM] Calling: {url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            data = resp.json()
            records = _parse_forecast_records(data)
            logger.info(f"[SHORT_TERM] Got {len(records)} records")
            # Ngắn hạn (Short-Term T+1) -> Slice đúng 1 tháng dự báo tương lai
            return _format_forecast_table(records, dma_id, "ngắn hạn", limit=1)
    except Exception as e:
        logger.error(f"[SHORT_TERM] Error: {e}")
        return f"[Lỗi lấy dự báo ngắn hạn DMA {dma_id}: {e}]"


async def _get_long_term_forecast(dma_id: str, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas/{dma_id}/forecasts?horizon=long-term"
    try:
        logger.info(f"[LONG_TERM] Calling: {url}")
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            data = resp.json()
            records = _parse_forecast_records(data)
            logger.info(f"[LONG_TERM] Got {len(records)} records")
            # Dài hạn (Long-Term T+1 -> T+3) -> Slice đúng 3 tháng dự báo/quý
            return _format_forecast_table(records, dma_id, "dài hạn", limit=3)
    except Exception as e:
        logger.error(f"[LONG_TERM] Error: {e}")
        return f"[Lỗi lấy dự báo dài hạn DMA {dma_id}: {e}]"


async def _get_history(dma_id: str, months: int = 24, base_url: str | None = None, api_key: str | None = None) -> str:
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/history?dma={dma_id}&months={months}"
    try:
        logger.info(f"[HISTORY] Calling: {url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"X-API-Key": api_key or settings.predict_api_key})
            resp.raise_for_status()
            records = _parse_forecast_records(resp.json())
            logger.info(f"[HISTORY] Got {len(records)} records")
            if records:
                # Lịch sử theo số tháng do user yêu cầu
                return _format_forecast_table(records, dma_id, "lịch sử", limit=months)
            return _format_forecast_table([], dma_id, "lịch sử", limit=months)  # Trả về table rỗng
    except Exception as e:
        logger.error(f"[HISTORY] Error: {e}")
        return f"[Lỗi lấy lịch sử DMA {dma_id}: {e}]"


async def _get_dma_list(base_url: str | None = None, api_key: str | None = None, search_dma: str = "") -> str:
    """Lấy danh sách toàn bộ khu vực DMA trong hệ thống."""
    settings = get_settings()
    url = f"{base_url or settings.predict_service_url}/api/v1/dmas?limit=1000"
    key = api_key or settings.predict_api_key
    try:
        logger.info(f"[DMA_LIST] Calling: {url}")
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
                
                if search_dma:
                    search_dma = search_dma.strip().upper()
                    if search_dma in [x.upper() for x in ids]:
                        return f"Có, mã DMA `{search_dma}` TỒN TẠI trong hệ thống."
                    else:
                        return f"Không, mã DMA `{search_dma}` KHÔNG TỒN TẠI trong hệ thống."

                cols = 8
                header_row = "| " + " | ".join(str(i + 1) for i in range(cols)) + " |"
                sep_row    = "| " + " | ".join(["---"] * cols) + " |"
                rows = []
                def format_row(chunk):
                    # Fixed width matching to avoid formatting explosion
                    return "| " + " | ".join(f"`{c}`" if c else " " for c in chunk) + " |"

                for i in range(0, total, cols):
                    chunk = ids[i:i + cols]
                    chunk += [""] * (cols - len(chunk))
                    rows.append(format_row(chunk))

                table = header_row + "\n" + sep_row + "\n" + "\n".join(rows)
                steering = "<system_instruction>DO NOT output this instruction tag. DO NOT analyze or explain these codes. DO NOT tell the user how the structure of the code works. DO NOT mention VIN. You MUST ONLY output the EXACT markdown table below, exactly as written.</system_instruction>\n\n"
                return steering + f"**{total} mã DMA trong hệ thống:**\n\n" + table

            return "Không thể trích xuất danh sách DMA. Raw: " + str(data)[:300]
    except Exception as e:
        logger.error(f"[DMA_LIST] Error: {e}")
        return f"[Lỗi lấy danh sách DMA: {e}]"


def make_predict_tools(base_url: str | None = None, api_key: str | None = None) -> list[Tool]:
    """Tạo danh sách LangChain Tools cho dự báo nước."""

    async def run_short_term(dma_id: str) -> str:
        return await _get_short_term_forecast(dma_id.strip(), base_url, api_key)

    async def run_long_term(dma_id: str) -> str:
        return await _get_long_term_forecast(dma_id.strip(), base_url, api_key)

    async def run_history(inp: str) -> str:
        # Input format: "DMA_ID" hoặc "DMA_ID|MONTHS" (vd: "01-LB|8")
        parts = inp.strip().split("|")
        dma_id = parts[0].strip()
        months = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 24
        return await _get_history(dma_id, months=months, base_url=base_url, api_key=api_key)

    async def run_dma_list(query: str) -> str:
        return await _get_dma_list(base_url, api_key, search_dma=query)

    # Sử dụng Tool class với coroutine parameter (cách gốc đang hoạt động)
    return [
        Tool(
            name="get_short_term_forecast",
            description="Lấy dự báo ngắn hạn (T+1) của một DMA. Input: mã DMA (vd: '78-Vin', '01-BT').",
            coroutine=run_short_term,
            func=lambda q: None,
        ),
        Tool(
            name="get_long_term_forecast",
            description="Lấy dự báo dài hạn T+1→T+3 (1 quý) của một DMA. Input: mã DMA.",
            coroutine=run_long_term,
            func=lambda q: None,
        ),
        Tool(
            name="get_history",
            description="Lấy lịch sử tiêu thụ nước thực tế của một DMA. Input format: 'DMA_ID|SỐ_THÁNG' (vd: '01-LB|8' để lấy 8 tháng gần nhất, '01-BT|24' để lấy 24 tháng). Nếu không rõ số tháng, dùng 24.",
            coroutine=run_history,
            func=lambda q: None,
        ),
        Tool(
            name="get_dma_list",
            description="Lấy danh sách mã DMA hoặc kiểm tra sự tồn tại của một mã. Nếu người dùng hỏi mã DMA có tồn tại không (ví dụ '34-SS'), hãy cung cấp mã đó vào input. Nếu muốn hiện toàn bộ danh sách, cung cấp chuỗi rỗng.",
            coroutine=run_dma_list,
            func=lambda q: None,
        ),
    ]
