r"""generate 가 멈춘 output 에서 검수자에게 넘길 QA 번들을 만든다.

generate 는 40,000장을 만든 뒤 manifests/fail_visual_qa.csv 와 함께 멈춘다. 검수자에게
40,000장 전체를 넘길 수는 없으므로, 표본으로 뽑힌 행의 이미지만 모아 한 폴더로 묶는다.

    python build_qa_bundle.py --output "D:\qf_full" --config ..\config.40k.json

산출:

    qa_bundle/
      fail_visual_qa.csv   비어 있는 판정 CSV (검수자가 채울 원본)
      images/              표본 이미지 사본
      review_tool.html     이미지가 내장된 자체 완결 검수 UI
      README.txt           판정 기준과 회수 절차

review_tool.html 은 외부 파일을 읽지 않는다. 이미지를 data URI 로 품고 있어서 파일
하나만 옮겨도 검수가 된다. 내보내는 CSV 는 generator.py 가 쓰는 11칸 스키마와 같으므로
merge_and_check.py 에 그대로 넣을 수 있다.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import shutil
from collections import defaultdict
from pathlib import Path

# generator.py::_visual_qa_gate 가 쓰는 컬럼과 정확히 같아야 한다.
QA_FIELDS = [
    "modality", "failure_case", "augmentation_subtype",
    "synthetic_id", "image_path", "source_filename", "source_image_path",
    "original_battery_id", "reviewer", "approved", "reason",
]

TEMPLATE = r"""<meta charset="utf-8">
<title>Visual QA 검수 — quality-fail-augment __VERSION__</title>
<style>
*{box-sizing:border-box}
body{font-family:'Malgun Gothic',system-ui,sans-serif;margin:0;background:#12141a;color:#e6e8ee}
header{position:sticky;top:0;z-index:20;background:#1a1d26;border-bottom:1px solid #2c3040;padding:10px 16px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0;font-weight:650}
input,select,button{font:inherit;background:#232735;color:#e6e8ee;border:1px solid #39405a;border-radius:6px;padding:6px 10px}
button{cursor:pointer}
button:hover{background:#2d3244}
button.primary{background:#3b6ef0;border-color:#3b6ef0;color:#fff;font-weight:600}
button.primary:hover{background:#2f5ed6}
.warn{color:#ffb020}.ok{color:#3ddc97}.bad{color:#ff5f6d}
.stats{display:flex;gap:16px;font-size:13px;flex-wrap:wrap;margin-top:8px}
.stats b{font-variant-numeric:tabular-nums}
main{padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:#1a1d26;border:2px solid #2c3040;border-radius:8px;padding:8px;cursor:pointer;transition:border-color .12s}
.card:hover{border-color:#4a5378}
.card.approved{border-color:#3ddc97}
.card.rejected{border-color:#ff5f6d}
.card.focus{outline:3px solid #3b6ef0;outline-offset:2px}
.card img{width:100%;display:block;border-radius:4px;background:#000;filter:var(--imgfilter,none)}
.meta{font-size:10px;color:#8b93ad;margin-top:6px;word-break:break-all;line-height:1.4}
.badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:99px;background:#39405a;color:#cdd3e6}
h2{font-size:15px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid #2c3040;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.rate{font-size:12px;font-weight:400}
#lightbox{position:fixed;inset:0;background:rgba(8,9,13,.97);z-index:50;display:none;flex-direction:column;align-items:center;gap:10px;padding:12px}
#lightbox.on{display:flex}
/* align-items:center 로 중앙 정렬하면 확대 시 넘친 위쪽에 스크롤이 닿지 않는다.
   margin:auto 방식은 작을 땐 중앙, 클 땐 양방향 스크롤이 모두 된다. */
#lbstage{flex:1;min-height:0;width:100%;display:flex;overflow:auto}
#lbimg{margin:auto;border-radius:6px;background:#000;transition:filter .1s;cursor:grab}
#lbimg.drag{cursor:grabbing}
#lbinfo{font-size:13px;color:#a8b0c8;text-align:center;line-height:1.6}
#lbtools{display:flex;gap:8px;align-items:center;font-size:12px;color:#8b93ad;flex-wrap:wrap;justify-content:center}
#lbtools button{padding:4px 10px;font-size:12px}
.tag{background:#232735;border:1px solid #39405a;border-radius:5px;padding:3px 8px;font-variant-numeric:tabular-nums}
.lbbtns{display:flex;gap:10px}
.lbbtns button{padding:10px 22px;font-size:15px;font-weight:600}
.hint{font-size:12px;color:#7a8199}
kbd{background:#2b3145;border:1px solid #454d6b;border-bottom-width:2px;border-radius:4px;padding:1px 6px;font-size:11px;font-family:inherit}
</style>

<header>
  <div class="bar">
    <h1>Visual QA 검수 __VERSION__</h1>
    <label>검수자 <input id="reviewer" placeholder="이름 입력" style="width:120px"></label>
    <label>케이스
      <select id="filter"><option value="">전체 (__TOTAL__)</option></select>
    </label>
    <button id="focusBtn">집중 검수 시작</button>
    <button id="cyc2">대비 (C)</button>
    <button class="primary" id="export">CSV 내보내기</button>
    <button id="reset">초기화</button>
  </div>
  <div class="stats">
    <span>판정 <b id="done">0</b>/__TOTAL__</span>
    <span class="ok">승인 <b id="nap">0</b></span>
    <span class="bad">거부 <b id="nrj">0</b></span>
    <span>__RATE_PCT__% 미달 케이스 <b id="nfail" class="warn">0</b></span>
    <span id="saveinfo" style="color:#7a8199"></span>
  </div>
</header>

<main id="main"></main>

<div id="lightbox">
  <div id="lbstage"><img id="lbimg"></div>
  <div id="lbtools">
    <button id="zOut">− 축소</button>
    <span class="tag" id="zLabel">100%</span>
    <button id="zIn">+ 확대</button>
    <button id="zFit">맞춤 (0)</button>
    <span style="width:12px"></span>
    <button id="cyc">대비 (C)</button>
    <span class="tag" id="cLabel">원본</span>
    <span style="width:12px"></span>
    <button id="invBtn">반전 (I)</button>
  </div>
  <div id="lbinfo"></div>
  <div class="lbbtns">
    <button class="ok" id="lbA">승인 (A)</button>
    <button class="bad" id="lbR">거부 (R)</button>
    <button id="lbSkip">건너뛰기 (Space)</button>
    <button id="lbClose">닫기 (Esc)</button>
  </div>
  <div style="font-size:12px;color:#8b93ad">
    <kbd>A</kbd> 승인 · <kbd>R</kbd> 거부 · <kbd>←</kbd><kbd>→</kbd> 이동 ·
    <kbd>C</kbd> 대비 · <kbd>I</kbd> 반전 · <kbd>+</kbd><kbd>−</kbd> 확대 · <kbd>Esc</kbd> 닫기
    <br><span style="color:#6b7390">확대 후에는 마우스로 끌어서 이동하거나 스크롤할 수 있습니다</span>
  </div>
</div>

<script>
const FIELDS = __FIELDS__;
const MIN_RATE = __MIN_RATE__;
const DATA = __DATA__;
const KEY = '__STORAGE_KEY__';

let state = JSON.parse(localStorage.getItem(KEY) || '{}');
let verdicts = state.verdicts || {};
let focusIdx = -1;

// CT 이미지는 어둡고 폭이 좁아(예: 53x512) 원본 대비로는 링/줄무늬가 묻힌다.
// 대비 프리셋과 확대를 두어 판정 정확도를 확보한다.
const FILTERS = [
  ['원본', ''],
  ['대비 +', 'brightness(1.25) contrast(1.8)'],
  ['대비 ++', 'brightness(1.5) contrast(2.6)'],
  ['최대', 'brightness(1.75) contrast(3.4)'],
];
let fIdx = 0, zoom = 0, invert = false;   // zoom 0 = 높이 맞춤

const $ = id => document.getElementById(id);
$('reviewer').value = state.reviewer || '';

function save(){
  localStorage.setItem(KEY, JSON.stringify({reviewer: $('reviewer').value, verdicts}));
  $('saveinfo').textContent = '자동 저장됨 ' + new Date().toLocaleTimeString('ko-KR');
}

const cases = [...new Set(DATA.map(d => d.mod + ' / ' + d.case))];
cases.forEach(c => {
  const o = document.createElement('option');
  o.value = c; o.textContent = c;
  $('filter').appendChild(o);
});

function visible(){
  const f = $('filter').value;
  return f ? DATA.filter(d => (d.mod + ' / ' + d.case) === f) : DATA;
}

function render(){
  const m = $('main');
  m.innerHTML = '';
  const groups = {};
  visible().forEach(d => {
    const k = d.mod + ' / ' + d.case;
    (groups[k] = groups[k] || []).push(d);
  });
  for (const [name, items] of Object.entries(groups)){
    const judged = items.filter(d => verdicts[d.id] !== undefined);
    const appr = judged.filter(d => verdicts[d.id] === true).length;
    const rate = judged.length ? appr / judged.length : 0;
    const h = document.createElement('h2');
    let cls = 'rate', txt = judged.length + '/' + items.length + ' 판정';
    if (judged.length){
      txt += ' · 승인율 ' + (rate*100).toFixed(1) + '%';
      cls += judged.length === items.length ? (rate >= MIN_RATE ? ' ok' : ' bad') : ' warn';
    }
    h.innerHTML = '<span>' + name + '</span><span class="' + cls + '">' + txt + '</span>';
    m.appendChild(h);
    const g = document.createElement('div');
    g.className = 'grid';
    items.forEach(d => {
      const c = document.createElement('div');
      c.className = 'card' + (verdicts[d.id] === true ? ' approved' : verdicts[d.id] === false ? ' rejected' : '');
      c.onclick = () => openBox(DATA.indexOf(d));
      const img = document.createElement('img');
      img.src = d.img; img.loading = 'lazy';
      c.appendChild(img);
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerHTML = '<span class="badge">' + (d.sub || '-') + '</span><br>' + d.id +
        '<br><span style="color:#6b7390">원본 ' + d.srcFile + '</span>';
      c.appendChild(meta);
      g.appendChild(c);
    });
    m.appendChild(g);
  }
  updateStats();
  save();
}

function updateStats(){
  const judged = DATA.filter(d => verdicts[d.id] !== undefined);
  $('done').textContent = judged.length;
  $('nap').textContent = judged.filter(d => verdicts[d.id] === true).length;
  $('nrj').textContent = judged.filter(d => verdicts[d.id] === false).length;
  const byCase = {};
  DATA.forEach(d => {
    const k = d.mod + '/' + d.case;
    (byCase[k] = byCase[k] || []).push(d);
  });
  let nfail = 0;
  for (const items of Object.values(byCase)){
    const j = items.filter(d => verdicts[d.id] !== undefined);
    if (j.length === items.length){
      const r = j.filter(d => verdicts[d.id] === true).length / j.length;
      if (r < MIN_RATE) nfail++;
    }
  }
  $('nfail').textContent = nfail;
}

function applyView(){
  const img = $('lbimg');
  const f = FILTERS[fIdx][1];
  const css = (f + (invert ? ' invert(1)' : '')).trim();
  img.style.filter = css;
  document.documentElement.style.setProperty('--imgfilter', css || 'none');
  $('cLabel').textContent = FILTERS[fIdx][0] + (invert ? ' + 반전' : '');
  if (zoom === 0){
    img.style.height = 'auto'; img.style.width = 'auto';
    img.style.maxHeight = '100%'; img.style.maxWidth = '100%';
    $('zLabel').textContent = '맞춤';
  } else {
    img.style.maxHeight = 'none'; img.style.maxWidth = 'none';
    img.style.width = (img.naturalWidth * zoom) + 'px';
    img.style.height = 'auto';
    $('zLabel').textContent = Math.round(zoom * 100) + '%';
  }
}

function openBox(i){
  focusIdx = i;
  const d = DATA[i];
  $('lbimg').src = d.img;
  $('lbinfo').innerHTML = '<b>' + d.mod + ' / ' + d.case + '</b> · ' + (d.sub || '-') +
    '<br>' + d.id + ' · ' + d.path +
    '<br><span style="color:#7a8199">원본 ' + d.srcFile + ' · battery ' + d.srcBattery + '</span>';
  $('lightbox').classList.add('on');
  applyView();
}

function closeBox(){ $('lightbox').classList.remove('on'); focusIdx = -1; }

function step(delta){
  const list = visible();
  if (!list.length) return;
  const cur = DATA[focusIdx];
  let i = list.indexOf(cur);
  i = (i + delta + list.length) % list.length;
  openBox(DATA.indexOf(list[i]));
}

function judge(v){
  if (focusIdx < 0) return;
  verdicts[DATA[focusIdx].id] = v;
  save(); updateStats();
  step(1);
  render();
  if (focusIdx >= 0) openBox(focusIdx);
}

$('focusBtn').onclick = () => {
  const list = visible();
  const next = list.find(d => verdicts[d.id] === undefined) || list[0];
  if (next) openBox(DATA.indexOf(next));
};
$('filter').onchange = render;
$('reviewer').oninput = save;
$('lbA').onclick = () => judge(true);
$('lbR').onclick = () => judge(false);
$('lbSkip').onclick = () => step(1);
$('lbClose').onclick = closeBox;
$('zIn').onclick = () => { zoom = zoom === 0 ? 1.5 : Math.min(zoom * 1.5, 12); applyView(); };
$('zOut').onclick = () => { zoom = zoom === 0 ? 1 : Math.max(zoom / 1.5, 0.25); applyView(); };
$('zFit').onclick = () => { zoom = 0; applyView(); };
$('invBtn').onclick = () => { invert = !invert; applyView(); };
$('cyc').onclick = $('cyc2').onclick = () => { fIdx = (fIdx + 1) % FILTERS.length; applyView(); render(); };

// 확대 상태에서 마우스로 끌어 이동한다.
(function(){
  const stage = $('lbstage'), img = $('lbimg');
  let dragging = false, sx = 0, sy = 0, sl = 0, st = 0;
  img.addEventListener('mousedown', e => {
    if (zoom === 0) return;
    dragging = true; sx = e.clientX; sy = e.clientY;
    sl = stage.scrollLeft; st = stage.scrollTop;
    img.classList.add('drag'); e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    stage.scrollLeft = sl - (e.clientX - sx);
    stage.scrollTop = st - (e.clientY - sy);
  });
  window.addEventListener('mouseup', () => { dragging = false; img.classList.remove('drag'); });
})();

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (!$('lightbox').classList.contains('on')) return;
  const k = e.key.toLowerCase();
  if (k === 'a') judge(true);
  else if (k === 'r') judge(false);
  else if (k === 'arrowright' || k === ' ') { step(1); e.preventDefault(); }
  else if (k === 'arrowleft') step(-1);
  else if (k === 'c') { fIdx = (fIdx + 1) % FILTERS.length; applyView(); }
  else if (k === 'i') { invert = !invert; applyView(); }
  else if (k === '+' || k === '=') { zoom = zoom === 0 ? 1.5 : Math.min(zoom * 1.5, 12); applyView(); }
  else if (k === '-') { zoom = zoom === 0 ? 1 : Math.max(zoom / 1.5, 0.25); applyView(); }
  else if (k === '0') { zoom = 0; applyView(); }
  else if (k === 'escape') closeBox();
});

$('reset').onclick = () => {
  if (!confirm('이 브라우저에 저장된 판정을 모두 지웁니다. 계속할까요?')) return;
  verdicts = {}; save(); render();
};

$('export').onclick = () => {
  const name = $('reviewer').value.trim();
  if (!name){ alert('검수자 이름을 먼저 입력하세요.'); return; }
  const judged = DATA.filter(d => verdicts[d.id] !== undefined);
  if (!judged.length){ alert('판정한 행이 없습니다.'); return; }
  const esc = s => /[",\n]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s;
  const lines = [FIELDS.join(',')];
  // 열 순서는 generator.py 의 fail_visual_qa.csv 와 같아야 merge_and_check.py 가 받는다.
  DATA.forEach(d => {
    const v = verdicts[d.id];
    lines.push([d.mod, d.case, d.sub, d.id, d.path, d.srcFile, d.srcPath, d.srcBattery,
      v === undefined ? '' : name,
      v === undefined ? '' : String(v),
      ''].map(x => esc(String(x))).join(','));
  });
  const blob = new Blob(['\ufeff' + lines.join('\r\n') + '\r\n'], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fail_visual_qa__' + name + '.csv';
  a.click();
};

applyView();
render();
</script>
"""

README = """quality-fail-augment {version} — Visual QA 검수 안내

이 폴더는 생성된 40,000장 중 FAIL 표본 {total}장을 검수하기 위한 것입니다.

무엇을 보는가
  "이 이미지가 해당 촬영 결함(failure case)으로 보이는가"만 판정합니다.
  픽셀 수치는 이미 자동 게이트가 걸렀으므로, 사람은 육안 인상만 봅니다.

판정 방법 (둘 중 하나)
  1) review_tool.html 을 브라우저로 엽니다. 이미지가 파일 안에 들어 있어 인터넷도,
     images 폴더도 필요 없습니다. 검수자 이름을 넣고 "집중 검수 시작"을 누른 뒤
     A(승인) / R(거부) 로 넘기면 됩니다. 판정은 브라우저에 자동 저장되므로 중간에
     닫아도 됩니다. 끝나면 "CSV 내보내기"를 눌러 파일을 받으십시오.
  2) fail_visual_qa.csv 를 직접 편집합니다. images/ 폴더의 파일 이름은
     "<synthetic_id>__<원래 파일명>" 형태이므로, CSV 의 synthetic_id 로 해당 이미지를
     찾을 수 있습니다. 그 이미지를 보면서 reviewer 와 approved 두 칸을 채웁니다.
     approved 에는 true 또는 false 를 씁니다. 행을 추가하거나 삭제하지 마십시오.

통과 기준
  케이스별 승인율 {rate_pct}% 이상. 케이스당 {per_case}장이므로 반려 1장까지는 통과하고,
  2장부터는 그 케이스 전체를 다시 만들어야 합니다.

주의
  CSV 는 UTF-8(BOM 포함)입니다. Excel 로 열어 저장해도 한글은 깨지지 않지만,
  다른 편집기로 저장할 때는 인코딩을 UTF-8 로 유지해 주십시오.

회수
  채운 CSV 를 담당자에게 전달하면 됩니다. 여러 명이 나눠 검수한 경우 각자의 CSV 를
  그대로 주십시오. review_tools/merge_and_check.py 가 병합과 승인율 사전 판정을
  대신 합니다.
"""


def _read_qa_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != QA_FIELDS:
            raise SystemExit(
                f"[{path}] 헤더가 계약과 다릅니다.\n  기대: {QA_FIELDS}\n  실제: {reader.fieldnames}"
            )
        return list(reader)


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="generate 가 만든 출력 폴더")
    parser.add_argument("--config", type=Path, required=True, help="생성에 쓴 config JSON")
    parser.add_argument("--bundle", type=Path, help="번들을 만들 위치 (기본: <output>-qa_bundle)")
    parser.add_argument("--version", default="v1.7", help="번들 표시용 버전 문자열")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    qa_csv = output / "manifests" / "fail_visual_qa.csv"
    if not qa_csv.is_file():
        raise SystemExit(f"QA CSV 가 없습니다. generate 를 먼저 끝내십시오: {qa_csv}")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    min_rate = float(config.get("visual_qa_min_approval_rate", 0.90))
    per_case = int(config.get("visual_qa_samples_per_case", 10))

    rows = _read_qa_rows(qa_csv)
    if not rows:
        raise SystemExit(f"QA CSV 에 행이 없습니다: {qa_csv}")

    bundle = (args.bundle or Path(f"{output}-qa_bundle")).resolve()
    images = bundle / "images"
    if bundle.exists():
        shutil.rmtree(bundle)
    images.mkdir(parents=True)

    data = []
    for index, row in enumerate(rows):
        source = output / row["image_path"]
        if not source.is_file():
            raise SystemExit(f"표본 이미지가 없습니다: {source}")
        # synthetic_id 를 앞에 붙여 사본 이름이 반드시 유일하게 한다. CSV 를 직접 편집하는
        # 검수자가 행과 파일을 synthetic_id 로 바로 맞출 수 있다는 이점도 있다.
        copy = images / f"{row['synthetic_id']}__{source.name}"
        if copy.exists():
            raise SystemExit(f"표본 사본 이름이 겹칩니다: {copy.name}")
        shutil.copy2(source, copy)
        data.append(
            {
                "i": index,
                "mod": row["modality"],
                "case": row["failure_case"],
                "sub": row["augmentation_subtype"],
                "id": row["synthetic_id"],
                "path": row["image_path"],
                "srcFile": row["source_filename"],
                "srcPath": row["source_image_path"],
                "srcBattery": row["original_battery_id"],
                "img": _data_uri(source),
            }
        )

    html = TEMPLATE
    for marker, value in (
        ("__VERSION__", args.version),
        ("__TOTAL__", str(len(rows))),
        ("__RATE_PCT__", f"{min_rate * 100:g}"),
        ("__MIN_RATE__", repr(min_rate)),
        ("__STORAGE_KEY__", f"qa_visual_{args.version.replace('.', '')}"),
        ("__FIELDS__", json.dumps(QA_FIELDS)),
        ("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))),
    ):
        html = html.replace(marker, value)
    (bundle / "review_tool.html").write_text(html, encoding="utf-8")

    shutil.copy2(qa_csv, bundle / "fail_visual_qa.csv")
    (bundle / "README.txt").write_text(
        README.format(
            version=args.version,
            total=len(rows),
            rate_pct=f"{min_rate * 100:g}",
            per_case=per_case,
        ),
        encoding="utf-8",
    )

    by_case: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        by_case[(row["modality"], row["failure_case"])] += 1
    print(f"번들: {bundle}")
    print(f"  표본 {len(rows)}장 · 케이스 {len(by_case)}개 · 승인 기준 {min_rate * 100:g}%")
    for key in sorted(by_case):
        print(f"    {key[0]:4} {key[1]:38} {by_case[key]:3}장")
    size = (bundle / "review_tool.html").stat().st_size
    print(f"  review_tool.html {size / 1_048_576:.1f} MB (이미지 내장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
