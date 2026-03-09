import asyncio
import os
import sqlite3
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Lấy settings từ api_gateway
from app.config import get_settings
from app.models.models import (
    Base, User, Workspace, WorkspaceUser, Document,
    Chat, RecoveryCode, ApiKey, EventLog, SystemSettings, DocumentStatus, UserRole
)

# Connect to source (SQLite)
SQLITE_DB_PATH = "./data/gateway.db"

async def migrate_data():
    print("=" * 50)
    print("   BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU TỪ SQLITE SANG POSTGRES   ")
    print("=" * 50)

    # 1. Khởi tạo engine Postgres và xóa sạch DB cũ nếu cần thiết, tạo bảng mới
    postgres_url = "postgresql+asyncpg://ocr_cuong:ocr_cuong@localhost:5432/api_gateway_db"
    pg_engine = create_async_engine(postgres_url, echo=False)
    
    async with pg_engine.begin() as conn:
        print("[+] Đang khởi tạo các bảng trên Postgres...")
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        
    async_session = sessionmaker(
        pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # 2. Đọc dữ liệu từ SQLite
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"[!] Không tìm thấy file SQLite tại {SQLITE_DB_PATH}")
        return
        
    print("[+] Đang đọc dữ liệu từ SQLite...")
    sl_conn = sqlite3.connect(SQLITE_DB_PATH)
    sl_conn.row_factory = sqlite3.Row
    sl_cur = sl_conn.cursor()

    from datetime import datetime

    def parse_dt(dt_str):
        if not dt_str:
            return None
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None

    async with async_session() as session:
        # 1. Migrate Users
        sl_cur.execute("SELECT * FROM users")
        users = sl_cur.fetchall()
        for u in users:
            new_user = User(
                id=u['id'],
                username=u['username'],
                email=u['email'],
                password_hash=u['password_hash'],
                role=UserRole(u['role'].lower() if u['role'] else 'default'),
                is_active=bool(u['is_active']),
                preferences=u['preferences'],
                created_at=parse_dt(u['created_at']),
                updated_at=parse_dt(u['updated_at'])
            )
            session.add(new_user)
        await session.commit()
        print(f"    - Đã đồng bộ {len(users)} users.")

        # 2. Migrate Workspaces
        sl_cur.execute("SELECT * FROM workspaces")
        workspaces = sl_cur.fetchall()
        for w in workspaces:
            new_ws = Workspace(
                id=w['id'],
                name=w['name'],
                slug=w['slug'],
                description=w['description'],
                llm_provider=w['llm_provider'],
                llm_model=w['llm_model'],
                embedding_model=w['embedding_model'],
                query_mode=w['query_mode'],
                owner_id=w['owner_id'],
                created_at=parse_dt(w['created_at']),
                updated_at=parse_dt(w['updated_at'])
            )
            session.add(new_ws)
        await session.commit()
        print(f"    - Đã đồng bộ {len(workspaces)} workspaces.")

        # 3. Migrate Workspace Users M-M table
        sl_cur.execute("SELECT * FROM workspace_users")
        wu = sl_cur.fetchall()
        for w in wu:
            new_wu = WorkspaceUser(
                id=w['id'],
                user_id=w['user_id'],
                workspace_id=w['workspace_id'],
                created_at=parse_dt(w['created_at'])
            )
            session.add(new_wu)
        await session.commit()
        print(f"    - Đã đồng bộ {len(wu)} workspace_users.")

    print("=" * 50)
    print("   HOÀN TẤT ĐỒNG BỘ SETUUP BẰNG TAY (MANUAL) THEO NEO4J   ")
    print("   Bỏ qua việc đồng bộ Lịch sử Chat và Documents !  ")
    print("==================================================")
    
    sl_conn.close()
    await pg_engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate_data())
