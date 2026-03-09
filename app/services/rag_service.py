"""
RAG Service client - kết nối tới rag-service backend.

Endpoints rag-service:
  POST /api/v1/chat          → Query + LLM response
  POST /api/v1/ocr-result    → Index tài liệu từ OCR JSON
"""
import httpx
from typing import Optional, Dict, Any, AsyncGenerator, List
import json

from app.config import get_settings

settings = get_settings()


class RAGServiceClient:
    """Client giao tiếp với rag-service backend."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.rag_service_url).rstrip("/")
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    async def health_check(self) -> bool:
        """Kiểm tra rag-service còn sống không."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def query(
        self,
        query: str,
        workspace_slug: str,
        mode: str = "consensus",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Gửi câu hỏi tới rag-service.
        Trả về {"answer": "...", "sources": [], "mode": "..."}
        """
        payload = {
            "messages": query,
            "mode": mode,
            "workspace": workspace_slug,
        }

        print(f"[RAG-CLIENT] Query → {self.base_url}/api/v1/chat | workspace={workspace_slug} | mode={mode}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            answer = data.get("response", data.get("answer", ""))
            return {
                "success": True,
                "answer": answer,
                "sources": data.get("sources", []),
                "images": data.get("images", []),
                "mode": data.get("mode", mode),
            }

        except Exception as e:
            print(f"[RAG-CLIENT] Query Error: {e}")
            return {
                "success": False,
                "answer": f"Lỗi kết nối RAG Service: {str(e)}",
                "sources": [],
                "images": [],
                "mode": mode,
            }

    async def query_stream(
        self,
        query: str,
        workspace_slug: str,
        mode: str = "mix"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream câu trả lời từ RAG system (real token streaming).
        Calls /api/v1/chat/stream trên rag-service và relay SSE events.
        Yields dict với key:
          - 'chunk' (str): token text từ LLM
          - 'sources' (list): sau khi hoàn tất
          - 'images' (list): ảnh liên quan
          - 'error' (str): nếu có lỗi
        """
        payload = {
            "messages": query,
            "mode": mode,
            "workspace": workspace_slug,
        }

        print(f"[RAG-CLIENT] Stream → {self.base_url}/api/v1/chat/stream | workspace={workspace_slug} | mode={mode}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/chat/stream",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except Exception:
                            continue

                        event_type = event.get("type")
                        if event_type == "token":
                            yield {"chunk": event.get("content", "")}
                        elif event_type == "done":
                            if event.get("sources"):
                                yield {"sources": event["sources"]}
                            if event.get("images"):
                                yield {"images": event["images"]}
                        elif event_type == "error":
                            yield {"error": event.get("content", "Unknown error")}
                            return

        except Exception as e:
            print(f"[RAG-CLIENT] Stream error: {e} – falling back to non-streaming")
            # Fallback về non-streaming nếu streaming endpoint lỗi
            result = await self.query(query=query, workspace_slug=workspace_slug, mode=mode)
            answer = result.get("answer", "")
            for i in range(0, len(answer), 15):
                yield {"chunk": answer[i:i + 15]}
            if result.get("sources"):
                yield {"sources": result["sources"]}
            if result.get("images"):
                yield {"images": result["images"]}

    async def ingest_ocr_result(
        self,
        ocr_json_data: Dict[str, Any],
        workspace_slug: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index tài liệu OCR vào rag-service.
        Gọi POST /api/v1/ocr-result với OCR JSON + workspace.
        """
        payload = {
            **ocr_json_data,
            "workspace": workspace_slug,
        }
        if job_id:
            payload["job_id"] = job_id

        print(f"[RAG-CLIENT] Ingest → {self.base_url}/api/v1/ingest/ocr-result | workspace={workspace_slug} | job_id={job_id}")

        ingest_timeout = httpx.Timeout(600.0, connect=10.0)
        async with httpx.AsyncClient(timeout=ingest_timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/ingest/ocr-result",
                json=payload,
            )
            response.raise_for_status()
            return response.json()


    async def process_document(
        self,
        file_path: str,
        workspace_slug: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upload file sang rag-service để xử lý luồng OCR -> Minio -> Indexing.
        """
        import os

        print(f"[RAG-CLIENT] process_document: sending {file_path} to RAG service upload endpoint")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        filename = os.path.basename(file_path)
        upload_timeout = httpx.Timeout(18000.0, read=18000.0, connect=10.0)
        
        async with httpx.AsyncClient(timeout=upload_timeout) as client:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                data = {"workspace": workspace_slug, "ocr_model": "deepseek"}
                resp = await client.post(
                    f"{self.base_url}/api/v1/ingest/upload",
                    files=files,
                    data=data
                )
            resp.raise_for_status()
            return resp.json()


    async def delete_workspace_data(self, workspace_slug: str) -> bool:
        """Gọi tới RAG service để dọn dẹp sạch sẽ Vector và Đồ thị (Neo4j) của workspace này."""
        print(f"[RAG-CLIENT] DELETE → {self.base_url}/api/v1/ingest/workspace/{workspace_slug}/purge_data")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/api/v1/ingest/workspace/{workspace_slug}/purge_data"
                )
                response.raise_for_status()
                data = response.json()
                print(f"[RAG-CLIENT] Purge Data Result: {data}")
                return True
        except Exception as e:
            print(f"[RAG-CLIENT] Lỗi khi báo RAG xoá dữ liệu cho {workspace_slug}: {e}")
            return False

    async def delete_document_data(self, workspace_slug: str, filename: str) -> bool:
        """Gọi tới RAG service để dọn dẹp sạch sẽ tài liệu ở cả Vector và Đồ thị."""
        print(f"[RAG-CLIENT] DELETE → {self.base_url}/api/v1/ingest/workspace/{workspace_slug}/document/{filename}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/api/v1/ingest/workspace/{workspace_slug}/document/{filename}"
                )
                response.raise_for_status()
                data = response.json()
                print(f"[RAG-CLIENT] Purge Document Data Result: {data}")
                return True
        except Exception as e:
            print(f"[RAG-CLIENT] Lỗi khi báo RAG xoá dữ liệu tài liệu {filename} cho {workspace_slug}: {e}")
            return False

    async def get_workspace_stats(self, workspace_slug: str) -> Dict[str, Any]:
        return {"documents": 0, "chunks": 0, "entities": 0}


# Singleton
rag_client = RAGServiceClient()


async def get_rag_client() -> RAGServiceClient:
    return rag_client
