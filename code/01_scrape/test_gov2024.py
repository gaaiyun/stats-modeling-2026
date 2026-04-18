import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9",
}
url = "https://www.gov.cn/gongbao/2024/issue_11246/202403/content_6941846.html"
r = requests.get(url, headers=HEADERS, timeout=30)
r.encoding = "utf-8"
print(f"HTTP {r.status_code}, {len(r.text)} bytes")
soup = BeautifulSoup(r.text, "lxml")

# Try various selectors
for sel in ["#UCAP-CONTENT", ".pages_content", ".content", "#content", "#zoom", ".article", ".article_con", "table"]:
    found = soup.select(sel)
    if found:
        text = found[0].get_text("\n", strip=True)
        print(f"\n[{sel}]: {len(text)} chars, first 300:\n{text[:300]}")

# Try table cells (公报 often uses tables)
tds = soup.find_all("td")
all_td_text = "\n".join(td.get_text(strip=True) for td in tds if len(td.get_text(strip=True)) > 100)
print(f"\n[ALL <td> text]: {len(all_td_text)} chars, first 500:\n{all_td_text[:500]}")
