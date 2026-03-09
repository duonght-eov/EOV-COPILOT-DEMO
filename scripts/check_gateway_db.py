import psycopg2
import os

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

print(f"\n{Colors.HEADER}{Colors.BOLD}=================================================={Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}   KIỂM TRA CƠ SỞ DỮ LIỆU POSTGRES (API GATEWAY)   {Colors.ENDC}")
print(f"{Colors.HEADER}{Colors.BOLD}=================================================={Colors.ENDC}")
print(f"PostgreSQL URL: postgresql://ocr_cuong:***@localhost:5432/api_gateway_db")

try:
    conn = psycopg2.connect(
        dbname="api_gateway_db",
        user="ocr_cuong",
        password="ocr_cuong",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    
    print(f"{Colors.OKGREEN}[+] Kết nối thành công!{Colors.ENDC}\n")
    
    # Lấy danh sách tất cả các bảng trong public schema
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    if not tables:
        print(f" ╰─ {Colors.WARNING}Không có bảng nào trong Database (Trống){Colors.ENDC}")
    else:
        print(f"{Colors.OKCYAN}{Colors.BOLD}--- THỐNG KÊ SỐ LƯỢNG BẢN GHI TỪNG BẢNG (API GATEWAY) ---{Colors.ENDC}")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print(f" ╰─ Bảng {Colors.WARNING}{table:<25}{Colors.ENDC} : {Colors.WARNING}0 rows{Colors.ENDC} (Trống)")
            else:
                print(f" ╰─ Bảng {Colors.OKGREEN}{table:<25}{Colors.ENDC} : {Colors.BOLD}{count} rows{Colors.ENDC}")
                
                # In thông tin thêm để User dễ hình dung
                if table == "workspaces":
                    cursor.execute(f"SELECT name, slug FROM {table}")
                    workspaces = cursor.fetchall()
                    for ws in workspaces:
                        print(f"    {Colors.OKBLUE}↳ Khách hàng: [{ws[1]}] - {ws[0]}{Colors.ENDC}")
                
                elif table == "users":
                    cursor.execute(f"SELECT username, role FROM {table}")
                    users = cursor.fetchall()
                    for user in users:
                        print(f"    {Colors.OKBLUE}↳ Tài khoản: {user[0]} (Role: {user[1]}){Colors.ENDC}")

    conn.close()
    print("")
    
except Exception as e:
    print(f"{Colors.FAIL}[-] LỖI KẾT NỐI POSTGRES: {e}{Colors.ENDC}\n")
