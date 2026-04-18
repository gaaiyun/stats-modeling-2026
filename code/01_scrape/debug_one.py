import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}
url = "https://www.gov.cn/premier/2017-03/16/content_5177940.htm"
r = requests.get(url, headers=HEADERS, timeout=30)
r.encoding = "utf-8"
print(f"status {r.status_code}, len {len(r.text)}")
soup = BeautifulSoup(r.text, "lxml")
divs = soup.find_all("div", id=True)
print("[divs with id]:", [(d.get("id"), len(d.get_text(strip=True))) for d in divs[:20]])
classes = {}
for d in soup.find_all("div", class_=True):
    cls = " ".join(d.get("class", []))
    classes.setdefault(cls, 0)
    classes[cls] += 1
print("[top div classes]:", sorted(classes.items(), key=lambda x: -x[1])[:15])
print("[len of all p text]:", sum(len(p.get_text(strip=True)) for p in soup.find_all("p")))
print("[first 500 char of all p text]:", "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))[:500])
print("\n=== body inner ===")
body = soup.body
print("body classes:", body.get("class") if body else None)
print("body text len:", len(body.get_text(strip=True)) if body else 0)
print("body text first 500:", body.get_text("\n", strip=True)[:500] if body else "")

idx = r.text.find("pages_content")
print(f"\npages_content index: {idx}")
if idx > 0:
    print(r.text[idx:idx + 3000])

