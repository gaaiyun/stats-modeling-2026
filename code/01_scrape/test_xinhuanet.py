import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9",
}
urls = {
    "新华2017": "http://www.xinhuanet.com/politics/2017lh/2017-03/16/c_1120638890.htm",
    "新华2020": "http://www.xinhuanet.com/politics/2020lh/2020-05/29/c_1126051808.htm",
    "新华2022": "http://www.news.cn/2022-03/12/c_1128464987.htm",
    "新华2023": "https://www.news.cn/2023-03/14/c_1129432017.htm",
    "公报2024": "https://www.gov.cn/gongbao/2024/issue_11246/202403/content_6941846.html",
}
for k, u in urls.items():
    r = requests.get(u, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")
    text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))
    print(f"\n=== {k}: HTTP {r.status_code}, p_text_len={len(text)} ===")
    print(text[:500])
