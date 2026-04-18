import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
}
urls = {
    2014: "https://www.gov.cn/guowuyuan/2014-03/14/content_2638989.htm",
    2017: "https://www.gov.cn/premier/2017-03/16/content_5177940.htm",
    "新华网2017": "http://www.xinhuanet.com/politics/2017lh/2017-03/16/c_1120638890.htm",
}
for k, u in urls.items():
    r = requests.get(u, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")
    body_text_len = len(soup.body.get_text(strip=True)) if soup.body else 0
    p_count = len(soup.find_all("p"))
    p_text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
    print(f"\n=== {k}: HTTP {r.status_code}, html {len(r.text)}, body text {body_text_len}, <p>={p_count}, p_text_len={len(p_text)} ===")
    print("First 200 of body text:", soup.body.get_text(strip=True)[:200] if soup.body else "(no body)")
