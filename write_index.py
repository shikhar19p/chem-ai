import os
out = "C:/Users/shikhar pulluri/Desktop/murali projject/frontend/index.html"
H = []
a = H.append

css = (
":root{--bg:#07111f;--surface:#0d1a2d;--card:#111e33;--card2:#162440;" +
"--border:#1a3356;--border2:#1f3d66;--accent:#22d3ee;--accent2:#818cf8;" +
"--warn:#fbbf24;--danger:#f87171;--text:#f1f5f9;--muted:#64748b;--muted2:#334155;" +
"--green:#34d399;--blue:#60a5fa;--purple:#a78bfa;--orange:#fb923c;--red:#f87171;--pink:#f472b6;--teal:#2dd4bf;}
" +
"*{box-sizing:border-box;margin:0;padding:0;}
" +
"body{background:var(--bg);color:var(--text);font-family:" + chr(39) + "Segoe UI" + chr(39) + ",system-ui,sans-serif;min-height:100vh;}
" +
"html{height:100%;width:100%;}
" +
"::-webkit-scrollbar{width:6px;height:6px;}" +
"::-webkit-scrollbar-track{background:var(--surface);}" +
"::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
"
)
a(css)
print("CSS ok")