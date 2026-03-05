"""
expose_ui.py
============
Chạy: python tests/expose_ui.py

- Tự download cloudflared nếu chưa có (~40MB, chỉ lần đầu)
- Tạo demo.html auto-login → workspace QTXD
- In link chia sẻ dạng https://xxx.trycloudflare.com/demo.html
- Ctrl+C để đóng
"""

import os, re, shutil, signal, subprocess, sys, time, urllib.request

UI_PORT       = int(os.getenv("UI_PORT",       "3000"))
DEMO_USERNAME = os.getenv("DEMO_USERNAME",      "admin")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD",      "admin123")
WORKSPACE     = os.getenv("WORKSPACE",          "qtxd")

G="\033[92m"; R="\033[91m"; C="\033[96m"; B="\033[1m"; E="\033[0m"

# Đường dẫn tới public/demo.html
PUBLIC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "../../speedmaint-ui/frontend/public"
))
DEMO_FILE = os.path.join(PUBLIC_DIR, "demo.html")

# Lưu cloudflared binary kế file này
CF_BIN = os.path.join(os.path.dirname(__file__), "cloudflared")
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

# ─── DEMO HTML ───────────────────────────────────────────────
DEMO_HTML = f"""<!DOCTYPE html><html lang="vi"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>EOVCopilot Demo</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0f1117;color:#e2e8f0;
min-height:100vh;display:flex;align-items:center;justify-content:center}}
.c{{text-align:center;padding:48px 40px;background:rgba(255,255,255,.04);
border:1px solid rgba(255,255,255,.08);border-radius:16px;max-width:380px;width:90%}}
h1{{font-size:20px;font-weight:600;margin-bottom:8px}}
p{{font-size:14px;color:#94a3b8;margin-bottom:28px}}
.s{{width:40px;height:40px;border:3px solid rgba(99,102,241,.2);
border-top-color:#6366f1;border-radius:50%;
animation:sp .8s linear infinite;margin:0 auto 18px}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
#st{{font-size:13px;color:#64748b}}
#er{{color:#f87171;font-size:14px;margin-top:14px;display:none}}</style>
</head><body><div class="c">
<div style="font-size:36px;margin-bottom:14px">🤖</div>
<h1>EOVCopilot Demo</h1>
<p>Hệ thống hỏi đáp thông minh<br>Vận hành &amp; Bảo trì CTXD</p>
<div class="s"></div>
<div id="st">Đang kết nối...</div>
<div id="er"></div></div>
<script>(async()=>{{
const T='anythingllm_authToken',U='anythingllm_user';
const st=m=>document.getElementById('st').textContent=m;
const er=m=>{{const e=document.getElementById('er');
e.textContent=m;e.style.display='block';
document.querySelector('.s').style.display='none';}};
try{{
st('Đang xác thực...');
const r=await fetch('/api/auth/login',{{method:'POST',
headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
body:'username={DEMO_USERNAME}&password={DEMO_PASSWORD}'}});
if(!r.ok)throw new Error('HTTP '+r.status);
const {{access_token}}=await r.json();
if(!access_token)throw new Error('Không nhận được token');
localStorage.setItem(T,access_token);
st('Đang tải workspace...');
try{{const me=await fetch('/api/auth/me',{{headers:{{'Authorization':'Bearer '+access_token}}}});
if(me.ok)localStorage.setItem(U,JSON.stringify(await me.json()));}}catch(_){{}}
st('Đang chuyển hướng...');
window.location.replace('/workspace/{WORKSPACE}');
}}catch(e){{er('❌ '+e.message);}}
}})();</script></body></html>"""


def setup_demo_html():
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    with open(DEMO_FILE, "w", encoding="utf-8") as f:
        f.write(DEMO_HTML)


def cleanup():
    try: os.remove(DEMO_FILE)
    except: pass


def get_cloudflared() -> str:
    """Trả về path đến cloudflared binary, download nếu chưa có."""
    # Kiểm tra trong PATH
    if shutil.which("cloudflared"):
        return "cloudflared"
    # Kiểm tra file local
    if os.path.exists(CF_BIN) and os.access(CF_BIN, os.X_OK):
        return CF_BIN
    # Download
    print(f"  {C}→{E} Đang tải cloudflared (~40MB, chỉ lần đầu)...")
    try:
        urllib.request.urlretrieve(CF_URL, CF_BIN)
        os.chmod(CF_BIN, 0o755)
        print(f"  {G}✓{E} Tải xong: {CF_BIN}")
        return CF_BIN
    except Exception as e:
        print(f"  {R}✗{E} Không tải được: {e}")
        return None


def run_cloudflared(cf_bin: str, port: int):
    """Chạy cloudflared, trả về (url, process). Nhanh ~3-5s."""
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://localhost:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = None
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    if not url:
        proc.terminate()
        return None, None
    return url, proc


def run_localhost_run(port: int):
    """Fallback: localhost.run qua SSH."""
    print(f"  {C}→{E} Đang kết nối localhost.run (fallback)...")
    proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-o", "ServerAliveInterval=20",
         "-o", "ServerAliveCountMax=3",
         "-R", f"80:localhost:{port}",
         "nokey@localhost.run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = None
    deadline = time.time() + 40
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r"https://[a-z0-9]+\.lhr\.life", line)
        if m:
            url = m.group(0)
            break
    if not url:
        proc.terminate()
        return None, None
    return url, proc


def print_link(demo_url: str):
    pad = 52
    print(f"""
{B}{C}╔{'═'*pad}╗{E}
{B}{C}║{E}  {'LINK CHIA SẺ CHO NGƯỜI TEST':<{pad-2}}{B}{C}  ║{E}
{B}{C}╠{'═'*pad}╣{E}
{B}{C}║{E}  {B}{G}{demo_url:<{pad-2}}{E}{B}{C}  ║{E}
{B}{C}╠{'═'*pad}╣{E}
{B}{C}║{E}  {'Bấm link → tự đăng nhập → workspace '+WORKSPACE:<{pad-2}}{B}{C}  ║{E}
{B}{C}║{E}  {'Nhấn Ctrl+C để đóng':<{pad-2}}{B}{C}  ║{E}
{B}{C}╚{'═'*pad}╝{E}
""")


def main():
    print(f"\n{B}{C}{'═'*56}{E}")
    print(f"{B}  EOVCopilot — Public Demo Link{E}")
    print(f"{B}{C}{'═'*56}{E}\n")

    # Tạo demo.html
    setup_demo_html()
    print(f"  {G}✓{E} demo.html ready")

    # Lấy cloudflared
    cf = get_cloudflared()
    url, proc = None, None

    if cf:
        print(f"  {C}→{E} Khởi cloudflared tunnel...")
        url, proc = run_cloudflared(cf, UI_PORT)

    # Fallback localhost.run
    if not url:
        url, proc = run_localhost_run(UI_PORT)

    if not url:
        print(f"\n  {R}✗{E} Không khởi được tunnel. Kiểm tra kết nối mạng.")
        cleanup()
        sys.exit(1)

    demo_url = f"{url}/demo.html"
    print_link(demo_url)

    def _exit(sig=None, frame=None):
        print("  Đang tắt...")
        if proc: proc.terminate()
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit)
    signal.signal(signal.SIGTERM, _exit)

    # Giữ tunnel sống — đọc output để SSH không timeout
    try:
        if proc:
            for _ in proc.stdout:
                pass
            proc.wait()
    except Exception:
        pass

    _exit()


if __name__ == "__main__":
    main()
