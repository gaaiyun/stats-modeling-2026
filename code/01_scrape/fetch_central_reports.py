"""
采集 2014-2025 年中央政府工作报告全文。
- 2014/2015/2016: gov.cn 旧模板（<p> 标签可解析）
- 2017-2019: gov.cn 内容由 JS 渲染，使用本地 WebFetch 缓存或新华网
- 2020-2025: 新华网/人民网（静态 HTML 包含 <p> 段落）
"""
from __future__ import annotations
import sys
import re
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from bs4 import BeautifulSoup
from configs.config import GOV_REPORT_DIR, LOGS_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

CENTRAL_URLS = {
    2014: "https://www.gov.cn/guowuyuan/2014-03/14/content_2638989.htm",
    2015: "https://www.gov.cn/guowuyuan/2015-03/16/content_2835101.htm",
    2016: "https://www.gov.cn/guowuyuan/2016-03/17/content_5054901.htm",
    2017: "http://www.xinhuanet.com/politics/2017lh/2017-03/16/c_1120638890.htm",
    2018: "http://www.xinhuanet.com/politics/2018lh/2018-03/22/c_1122575588.htm",
    2019: "http://www.xinhuanet.com/politics/2019lh/2019-03/16/c_1124242390.htm",
    2020: "http://www.xinhuanet.com/politics/2020lh/2020-05/29/c_1126051808.htm",
    2021: "http://www.news.cn/politics/2021lh/2021-03/12/c_1127205339.htm",
    2022: "http://www.news.cn/2022-03/12/c_1128464987.htm",
    2023: "https://www.news.cn/2023-03/14/c_1129432017.htm",
    2024: "http://www.news.cn/20240312/a4bc7208e1f046199a7fcb9e6bfdfa59/c.html",
    2025: "http://lianghui.people.com.cn/2025/n1/2025/0312/c460142-40437673.html",
}


def clean_text(text: str) -> str:
    text = re.sub(r"[\u3000\xa0]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    return text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    candidates = (
        soup.select("#UCAP-CONTENT")
        or soup.select(".pages_content")
        or soup.select(".article_con")
        or soup.select(".article")
        or soup.select("#content")
        or soup.select("#zoom")
    )
    if candidates:
        text = candidates[0].get_text("\n", strip=True)
    else:
        ps = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        text = "\n".join(ps)
    return clean_text(text)


def fetch_one(year: int, url: str, session: requests.Session) -> str | None:
    print(f"  >> {year}: {url}")
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
        r.encoding = "utf-8"
        if r.status_code != 200:
            print(f"     HTTP {r.status_code}")
            return None
        text = extract_text(r.text)
        if len(text) < 1500:
            print(f"     WARN too short ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"     EXC: {e}")
        return None


def main():
    out_dir = GOV_REPORT_DIR / "central"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {}
    s = requests.Session()
    for year, url in sorted(CENTRAL_URLS.items()):
        path = out_dir / f"中央_{year}.txt"
        if path.exists() and path.stat().st_size > 5000:
            print(f"[skip] {path.name} ({path.stat().st_size} bytes)")
            log[year] = {"status": "cached", "size": path.stat().st_size, "url": url}
            continue
        text = fetch_one(year, url, s)
        if text and len(text) >= 1500:
            path.write_text(text, encoding="utf-8")
            log[year] = {"status": "ok", "size": len(text), "url": url}
        else:
            log[year] = {"status": "fail", "size": len(text) if text else 0, "url": url}
        time.sleep(1.2)

    log_path = LOGS_DIR / "central_scrape_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for v in log.values() if v["status"] in ("ok", "cached"))
    print(f"\n=== 中央政府工作报告采集完成: {ok}/{len(CENTRAL_URLS)} 篇成功 ===")
    print(f"日志：{log_path}")


if __name__ == "__main__":
    main()
