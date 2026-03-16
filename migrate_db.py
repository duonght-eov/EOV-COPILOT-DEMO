import asyncio
import os
import sys

# Thêm đường dẫn app vào sys.path để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app.models import Workspace, WorkspaceConnector, ConnectorEndpoint
from sqlalchemy import text

async def upgrade_db():
    print("Bắt đầu nâng cấp CSDL...")
    
    # Block 1
    async with engine.begin() as conn:
        try:
            print("Thêm cột is_predict_enabled...")
            await conn.execute(text("ALTER TABLE workspaces ADD COLUMN is_predict_enabled BOOLEAN DEFAULT true"))
        except Exception as e:
            print(f"Cột is_predict_enabled có thể đã tồn tại.")
            
    # Block 2
    async with engine.begin() as conn:
        try:
            print("Thêm cột predict_llm_model...")
            await conn.execute(text("ALTER TABLE workspaces ADD COLUMN predict_llm_model VARCHAR(100)"))
        except Exception as e:
            print(f"Cột predict_llm_model có thể đã tồn tại.")

    # Block 3
    async with engine.begin() as conn:
        print("Tạo các bảng mới (WorkspaceConnector, ConnectorEndpoint)...")
        await conn.run_sync(Base.metadata.create_all)
        print("Cấu trúc CSDL đã được nâng cấp thành công!")
        
    # 3. Kích hoạt tính năng dự đoán và cấu hình connection mẫu cho speedmaint-predict
    from app.database import async_session_maker
    from sqlalchemy import select
    
    async with async_session_maker() as db:
        result = await db.execute(select(Workspace).where(Workspace.slug == "speedmaint-predict"))
        workspace = result.scalar_one_or_none()
        
        if workspace:
            print(f"Đã tìm thấy Workspace {workspace.slug}, ID: {workspace.id}")
            workspace.is_predict_enabled = True
            
            # Kiểm tra xem đã có connector chưa
            result_conn = await db.execute(select(WorkspaceConnector).where(WorkspaceConnector.workspace_id == workspace.id))
            connector = result_conn.scalar_one_or_none()
            
            if not connector:
                print("Đang tạo Connector mẫu cho máy 10.0.0.62...")
                new_connector = WorkspaceConnector(
                    workspace_id=workspace.id,
                    name="Máy chủ Dự báo Hanoi Water",
                    connector_type="REST_API",
                    base_url="http://10.0.0.62:8000",
                    auth_type="none"
                )
                db.add(new_connector)
            
            await db.commit()
            print("Đã cập nhật dữ liệu mẫu cho speedmaint-predict thành công!")
        else:
            print("Không tìm thấy workspace 'speedmaint-predict'.")

if __name__ == "__main__":
    asyncio.run(upgrade_db())
