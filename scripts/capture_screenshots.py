import subprocess
import time
import json
import base64
import urllib.request
import asyncio
import websockets
import os

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = os.path.abspath("docs/images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def capture_page(target_url, output_filename, window_size="1440,1100", scroll_y=0, wait_seconds=8):
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--window-size={window_size}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        target_url
    ])
    time.sleep(2)

    try:
        req = urllib.request.urlopen("http://localhost:9222/json/list")
        tabs = json.loads(req.read().decode('utf-8'))
        page_tabs = [t for t in tabs if t.get("type") == "page"]
        ws_url = page_tabs[0]['webSocketDebuggerUrl']

        async with websockets.connect(ws_url, max_size=25_000_000) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await ws.recv()

            print(f"Waiting {wait_seconds}s for {target_url} to render...")
            await asyncio.sleep(wait_seconds)

            if scroll_y > 0:
                print(f"Scrolling by {scroll_y}px...")
                scroll_js = f"""
                (() => {{
                    window.scrollTo(0, {scroll_y});
                    document.querySelectorAll('*').forEach(el => {{
                        if (el.scrollHeight > el.clientHeight && el.clientHeight > 200) {{
                            el.scrollTop = {scroll_y};
                        }}
                    }});
                }})()
                """
                await ws.send(json.dumps({
                    "id": 2,
                    "method": "Runtime.evaluate",
                    "params": {"expression": scroll_js}
                }))
                await asyncio.sleep(2)

            # Capture screenshot
            await ws.send(json.dumps({
                "id": 3,
                "method": "Page.captureScreenshot",
                "params": {"format": "png"}
            }))

            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("id") == 3:
                    img_data = base64.b64decode(data["result"]["data"])
                    filepath = os.path.join(OUTPUT_DIR, output_filename)
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    print(f"Saved {filepath} ({len(img_data)} bytes)")
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

async def main():
    # Overview page: window size 1440x1550 captures KPI cards, heat strip, headline finding banner, and donut/distribution charts
    await capture_page("http://localhost:8501", "dashboard_overview.png", window_size="1440,1550", scroll_y=0, wait_seconds=8)
    await asyncio.sleep(2)
    # Model Performance page: scroll down to capture threshold comparison cards and confusion matrix heatmap
    await capture_page("http://localhost:8501/Model_Performance", "dashboard_model_performance.png", window_size="1440,1100", scroll_y=1100, wait_seconds=8)

if __name__ == "__main__":
    asyncio.run(main())
