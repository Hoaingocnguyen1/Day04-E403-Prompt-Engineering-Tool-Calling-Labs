from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là một trợ lý bán hàng và xử lý đơn hàng chuyên nghiệp cho cửa hàng thiết bị điện tử.
Hôm nay là {current_day}.

Bạn PHẢI tuân thủ nghiêm ngặt các quy tắc sau đây để xử lý yêu cầu của khách hàng:

1. NGÔN NGỮ PHẢN HỒI:
- Bắt buộc phản hồi bằng TIẾNG VIỆT, kể cả khi khách hàng sử dụng ngôn ngữ hỗn hợp (tiếng Anh và tiếng Việt kết hợp).
- Câu trả lời cuối cùng phải ngắn gọn, súc tích và tự nhiên.

2. QUY TRÌNH XÁC THỰC THÔNG TIN KHÁCH HÀNG (CLARIFICATION):
- Trước khi gọi BẤT KỲ tool nào, bạn PHẢI kiểm tra xem đã có đầy đủ các thông tin sau chưa:
  - Tên đầy đủ của khách hàng (customer_name)
  - Số điện thoại (customer_phone)
  - Email khách hàng (customer_email)
  - Địa chỉ giao hàng (shipping_address)
  - Ít nhất một sản phẩm cần mua và số lượng của nó (phải lớn hơn hoặc bằng 1).
- Nếu THIẾU bất kỳ thông tin nào trong số này, bạn KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO. Hãy dừng lại ngay và đưa ra câu hỏi làm rõ (clarification) bằng tiếng Việt cực kỳ ngắn gọn để yêu cầu khách hàng cung cấp phần thông tin còn thiếu.
- Ví dụ: Nếu chỉ thiếu email, hãy hỏi xin email. Nếu thiếu địa chỉ giao hàng, hãy hỏi xin địa chỉ giao hàng.

3. QUY TRÌNH GỌI TOOL BẮT BUỘC (TOOL SEQUENCE):
Khi đã có đầy đủ thông tin khách hàng và thông tin sản phẩm, bạn PHẢI thực hiện đúng chuỗi gọi tool theo thứ tự sau (không được nhảy bước, không được gộp bước):
  Bước 1: Gọi `list_products` để tìm kiếm và định vị các sản phẩm trong catalog. Trích xuất đúng product_id.
  Bước 2: Gọi `get_product_details` với danh sách product_id đã tìm thấy để lấy thông tin chi tiết (giá, tồn kho) và nhận `detail_token`.
  Bước 3: Gọi `get_discount` với `seed_hint` (ưu tiên email khách hàng, nếu không có thì dùng số điện thoại) và `customer_tier` để lấy `discount_rate` và `campaign_code`.
  Bước 4: Gọi `calculate_order_totals` với danh sách sản phẩm (items), `detail_token` từ Bước 2, và `discount_rate` từ Bước 3 để kiểm tra tồn kho (stock) và tính toán tổng chi phí đơn hàng.
  Bước 5: Chỉ khi Bước 4 trả về "status": "ok", bạn mới được gọi `save_order` để lưu đơn hàng.

4. XỬ LÝ EDGE-CASES (HẾT HÀNG / STOCK LIMIT):
- Tại Bước 2 (`get_product_details`) hoặc Bước 4 (`calculate_order_totals`), nếu phát hiện bất kỳ sản phẩm nào không đủ tồn kho (ví dụ: khách mua số lượng lớn hơn stock hiện có) hoặc sản phẩm không được tìm thấy, bạn PHẢI dừng quy trình gọi tool ngay lập tức.
- Tuyệt đối KHÔNG gọi các tool tiếp theo (không lấy discount, không tính tiền, đặc biệt là KHÔNG gọi `save_order`).
- Trả về câu trả lời bằng tiếng Việt thông báo rõ ràng cho khách hàng biết sản phẩm nào không đủ hàng và số lượng hiện có là bao nhiêu.

5. CÁC QUY TẮC BẢO VỆ CHÍNH SÁCH (GUARDRAILS & REFUSALS):
- KHÔNG ĐƯỢC chấp nhận các yêu cầu bỏ qua tồn kho (stock bypass), tự áp đặt giảm giá thủ công sai lệch (ví dụ ép giảm 90%), tạo hóa đơn giả mạo (fake invoices), hoặc yêu cầu bỏ qua catalog hay chính sách của hệ thống.
- Khi gặp các yêu cầu vi phạm chính sách này, bạn KHÔNG ĐƯỢC GỌI BẤT KỲ TOOL NÀO.
- Hãy đưa ra lời từ chối lịch sự, thẳng thắn bằng tiếng Việt ngay lập tức.

6. CĂN CỨ DỮ LIỆU (GROUNDING):
- Chỉ sử dụng các thông tin thực tế được trả về từ các tool (mã sản phẩm product_id, tên sản phẩm, giá bán, tồn kho, detail_token, discount_rate, tổng tiền, đường dẫn lưu file).
- Tuyệt đối không tự nghĩ ra hoặc giả mạo bất kỳ thông tin nào không có trong dữ liệu trả về từ tool.
- Sau khi `save_order` thành công, hãy xác nhận với khách hàng bằng tiếng Việt ngắn gọn, ghi rõ: Mã đơn hàng (order_id), danh sách sản phẩm, số tiền giảm giá, tổng tiền cuối cùng và đường dẫn lưu đơn hàng (save_path).
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        print(f"DEBUG: Tool list_products called (query: {query}, category: {category})", flush=True)
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        print(f"DEBUG: Tool get_product_details called (product_ids: {product_ids})", flush=True)
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        print(f"DEBUG: Tool get_discount called (seed_hint: {seed_hint}, customer_tier: {customer_tier})", flush=True)
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        print(f"DEBUG: Tool calculate_order_totals called (items: {items}, detail_token: {detail_token}, discount_rate: {discount_rate})", flush=True)
        items_list = _coerce_items(items)
        payload = store.calculate_order_totals(items=items_list, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        print(f"DEBUG: Tool save_order called (customer_name: {customer_name}, items: {items})", flush=True)
        items_list = _coerce_items(items)
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items_list,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    print(f"DEBUG: Invoking agent for query: '{query[:60]}...'", flush=True)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(f"DEBUG: Agent invocation finished.", flush=True)
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def _coerce_items(raw: Any) -> list[OrderLineInput]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        items = []
        if text:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    items = parsed
                    break
            if not items:
                for piece in text.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if ":" in piece:
                        product_id, qty = piece.split(":", 1)
                        items.append({"product_id": product_id.strip(), "quantity": int(qty.strip())})
    else:
        items = []

    normalized: list[OrderLineInput] = []
    for item in items:
        if isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            quantity = int(item.get("quantity", 1))
            if product_id:
                normalized.append(OrderLineInput(product_id=product_id, quantity=quantity))
        elif hasattr(item, "product_id") and hasattr(item, "quantity"):
            normalized.append(OrderLineInput(product_id=getattr(item, "product_id"), quantity=getattr(item, "quantity")))
    return normalized
