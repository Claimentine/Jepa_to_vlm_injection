import json
import os

OUT_DIR = os.environ.get("EGODEX_PROBE_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(OUT_DIR, "kinematic_signals.json")
OUT_PATH = os.path.join(OUT_DIR, "kinematic_signal_check.html")

with open(DATA_PATH) as f:
    data = json.load(f)

W, H = 640, 130
PAD_L, PAD_R, PAD_T, PAD_B = 34, 8, 8, 18


def scale_x(i, n):
    return PAD_L + (W - PAD_L - PAD_R) * (i / max(n - 1, 1))


def scale_y(v, vmin, vmax):
    if vmax <= vmin:
        vmax = vmin + 1e-6
    frac = (v - vmin) / (vmax - vmin)
    return H - PAD_B - frac * (H - PAD_T - PAD_B)


def build_card(key, ep):
    n = ep["T"]
    aperture = ep["primary_aperture"]
    state = ep["primary_grasp_state"]
    contact = ep["contact_candidate"]
    active_hand = ep["active_hand"]
    vmin, vmax = min(aperture), max(aperture)

    bands = []
    i = 0
    while i < n:
        s = state[i]
        j = i
        while j < n and state[j] == s:
            j += 1
        if s != 0:
            x0 = scale_x(i, n)
            x1 = scale_x(min(j, n - 1), n)
            cls = "band-open" if s == 1 else "band-close"
            bands.append(
                f'<rect class="{cls}" x="{x0:.1f}" y="{PAD_T}" '
                f'width="{max(x1 - x0, 1):.1f}" height="{H - PAD_T - PAD_B}"></rect>'
            )
        i = j

    # active-hand ticks: small marks along the bottom axis, colored by hand
    hand_ticks = []
    i = 0
    while i < n:
        h = active_hand[i]
        j = i
        while j < n and active_hand[j] == h:
            j += 1
        x0 = scale_x(i, n)
        x1 = scale_x(min(j, n - 1), n)
        cls = "hand-R" if h == 1 else "hand-L"
        hand_ticks.append(
            f'<rect class="{cls}" x="{x0:.1f}" y="{H-PAD_B+2}" '
            f'width="{max(x1 - x0, 1):.1f}" height="4"></rect>'
        )
        i = j

    pts = " ".join(
        f"{scale_x(i, n):.1f},{scale_y(v, vmin, vmax):.1f}" for i, v in enumerate(aperture)
    )

    contact_marks = []
    for i, c in enumerate(contact):
        if c:
            x = scale_x(i, n)
            y = scale_y(aperture[i], vmin, vmax)
            contact_marks.append(
                f'<path class="contact-mark" d="M{x:.1f},{y-9:.1f} l4,7 l-8,0 z"></path>'
            )

    meta = ep["meta"]
    title = f"{key}"
    n_right = sum(1 for a in active_hand if a == 1)
    n_left = n - n_right
    subtitle = (
        f"task: {meta['task']} · objects: {', '.join(meta['llm_objects']) or '—'} · "
        f"verbs: {', '.join(meta['llm_verbs']) or '—'} · T={n} · active R/L frames={n_right}/{n_left}"
    )

    svg = f'''
    <svg viewBox="0 0 {W} {H+6}" class="chart-svg" data-key="{key}">
      <line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" y2="{H-PAD_B}" class="axis-line"></line>
      <text x="{PAD_L}" y="{PAD_T+8}" class="axis-label">{vmax:.2f}</text>
      <text x="{PAD_L}" y="{H-PAD_B-2}" class="axis-label">{vmin:.2f}</text>
      {''.join(bands)}
      <polyline points="{pts}" class="aperture-line"></polyline>
      {''.join(contact_marks)}
      {''.join(hand_ticks)}
    </svg>
    '''

    return f'''
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-subtitle">{subtitle}</div>
      {svg}
    </div>
    '''


cards = "\n".join(build_card(k, v) for k, v in data.items())

html = f'''<title>EgoDex 抓握运动学信号 · 信号质量检查(动态主手)</title>
<style>
.viz-root {{
  --surface-1: #fcfcfb;
  --surface-2: #f4f3f0;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #8a887f;
  --border: #e4e2dc;
  --series-blue: #2a78d6;
  --series-red: #e34948;
  --band-open: rgba(42,120,214,0.16);
  --band-close: rgba(227,73,72,0.16);
  --critical: #d03b3b;
  --hand-r: #4a3aa7;
  --hand-l: #eda100;
}}
@media (prefers-color-scheme: dark) {{
  .viz-root {{
    --surface-1: #1a1a19; --surface-2: #232322; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #8a887f; --border: #34332f;
    --series-blue: #3987e5; --series-red: #e66767;
    --band-open: rgba(57,135,229,0.20); --band-close: rgba(230,103,103,0.20);
    --critical: #e66767; --hand-r: #9085e9; --hand-l: #c98500;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1: #1a1a19; --surface-2: #232322; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --text-muted: #8a887f; --border: #34332f;
  --series-blue: #3987e5; --series-red: #e66767;
  --band-open: rgba(57,135,229,0.20); --band-close: rgba(230,103,103,0.20);
  --critical: #e66767; --hand-r: #9085e9; --hand-l: #c98500;
}}
:root[data-theme="light"] .viz-root {{
  --surface-1: #fcfcfb; --surface-2: #f4f3f0; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #8a887f; --border: #e4e2dc;
  --series-blue: #2a78d6; --series-red: #e34948;
  --band-open: rgba(42,120,214,0.16); --band-close: rgba(227,73,72,0.16);
  --critical: #d03b3b; --hand-r: #4a3aa7; --hand-l: #eda100;
}}
.viz-root {{
  background: var(--surface-1); color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 24px; max-width: 1400px; margin: 0 auto;
}}
.viz-header h1 {{ font-size: 18px; margin: 0 0 4px; }}
.viz-header p {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 4px; line-height: 1.5; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 14px 0 20px; font-size: 12px; color: var(--text-secondary); }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 14px; height: 10px; border-radius: 2px; display: inline-block; }}
.legend-line {{ width: 16px; height: 2px; background: var(--series-blue); display: inline-block; }}
.legend-tri {{ width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 8px solid var(--critical); display:inline-block; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(620px, 1fr)); gap: 16px; }}
.card {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }}
.card-title {{ font-size: 13px; font-weight: 600; margin-bottom: 2px; }}
.card-subtitle {{ font-size: 11.5px; color: var(--text-muted); margin-bottom: 8px; }}
.chart-svg {{ width: 100%; height: auto; display: block; }}
.axis-line {{ stroke: var(--border); stroke-width: 1; }}
.axis-label {{ font-size: 9px; fill: var(--text-muted); }}
.aperture-line {{ fill: none; stroke: var(--series-blue); stroke-width: 2; }}
.band-open {{ fill: var(--band-open); }}
.band-close {{ fill: var(--band-close); }}
.contact-mark {{ fill: var(--critical); }}
.hand-R {{ fill: var(--hand-r); }}
.hand-L {{ fill: var(--hand-l); }}
</style>
<div class="viz-root">
  <div class="viz-header">
    <h1>EgoDex 手部运动学信号检查(动态主手选择,每类任务取 1 个 episode)</h1>
    <p>信号:每帧的"主操作手"抓握开合度(拇指尖↔食指尖距离,按手掌宽度归一化)。主手由局部窗口内两手的活动强度(开合度局部方差 + 手部速度)动态选出,不再固定用右手。蓝色背景 = 张开中,红色背景 = 闭合中,无底色 = 稳定。三角 = 启发式接触候选帧。底部细条:紫色 = 该帧主手为右手,黄色 = 左手。</p>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="legend-line"></span> 主手抓握开合度(归一化)</div>
    <div class="legend-item"><span class="legend-swatch" style="background:var(--band-open)"></span> 张开中</div>
    <div class="legend-item"><span class="legend-swatch" style="background:var(--band-close)"></span> 闭合中</div>
    <div class="legend-item"><span class="legend-tri"></span> 接触候选帧</div>
    <div class="legend-item"><span class="legend-swatch" style="background:var(--hand-r)"></span> 主手=右</div>
    <div class="legend-item"><span class="legend-swatch" style="background:var(--hand-l)"></span> 主手=左</div>
  </div>
  <div class="grid">
    {cards}
  </div>
</div>
'''

with open(OUT_PATH, "w") as f:
    f.write(html)

print("wrote", OUT_PATH)
