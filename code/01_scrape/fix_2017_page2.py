"""补 2017 报告第 2 页内容（新华网用 _2.htm 分页）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import re
import requests
from bs4 import BeautifulSoup
from configs.config import GOV_REPORT_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

URL_BASE = "http://www.xinhuanet.com/politics/2017lh/2017-03/16/c_1120638890"

def extract(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    ps = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    return "\n".join(ps)


texts = []
for page in [2, 3, 4, 5, 6]:
    url = f"{URL_BASE}_{page}.htm"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    print(f"page {page}: HTTP {r.status_code}, html {len(r.text)}")
    if r.status_code != 200:
        break
    text = extract(r.text)
    print(f"  extracted {len(text)} chars; head: {text[:80]}")
    if len(text) < 500:
        break
    texts.append(text)

if texts:
    out = GOV_REPORT_DIR / "central" / "中央_2017.txt"
    existing = out.read_text(encoding="utf-8")
    existing = re.sub(r"点击查看专题.*$", "", existing, flags=re.S).rstrip()
    merged = existing + "\n" + "\n".join(texts)
    out.write_text(merged, encoding="utf-8")
    print(f"\n[done] 2017 现在: {len(merged)} 字符")
