"""Build index.html (GitHub Pages site) from report/report.html.

Usage: py report/build_site.py   (run from the repository root)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/chhelu123/time-series-forecasting"

html = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
html = html.replace('src="../outputs/', 'src="outputs/').replace('href="../outputs/', 'href="outputs/')

screen_css = """
@media screen {
  html { background:#f4f4f1; }
  body { max-width:900px; margin:0 auto; padding:0 24px 48px; background:#fff; font-size:16px;
         box-shadow:0 0 24px rgba(0,0,0,.06); }
  .cover { height:auto; padding:48px 0 32px; border-bottom:1px solid var(--rule); }
  .cover tr:has(td.blank) { display:none; }
  .links { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin-top:4px; }
  .links a { border:1px solid var(--accent); border-radius:6px; padding:8px 14px; word-break:normal; }
  .links a.primary { background:var(--accent); color:#fff; }
  main table { max-width:100%; display:block; overflow-x:auto; }
  pre { overflow-x:auto; }
  figure img { max-height:none; }
}
@media screen and (max-width:600px) {
  body { padding:0 16px 32px; font-size:15px; }
  .cover h1 { font-size:24px; }
  p, li { text-align:left; }
}
@media print { .links { display:none; } }
"""
links = f"""<div class="links">
<a class="primary" href="REPORT.pdf">Download report (PDF)</a>
<a href="{REPO}/blob/main/time_series_analysis.ipynb">Jupyter notebook</a>
<a href="{REPO}/blob/main/src/analysis.py">Python script</a>
<a href="{REPO}">Source code on GitHub</a>
</div>"""

html = html.replace("</style>", screen_css + "</style>", 1)
html = html.replace("</section>\n<main>", links + "\n</section>\n<main>", 1)
assert links in html, "cover/main boundary not found"

(ROOT / "index.html").write_text(html, encoding="utf-8")
(ROOT / ".nojekyll").write_text("", encoding="utf-8")
print("wrote index.html")
