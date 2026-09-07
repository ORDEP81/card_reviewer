"""Regenerate label.html, and serve it.

    python3 training/photos/make_labeller.py            # regenerate only
    python3 training/photos/make_labeller.py --serve    # regenerate + serve

Serve it rather than opening the file directly. Browsers block a file://
page from reading other local files, so the images come back as "could not
load" even though the paths are correct — which looks exactly like a broken
page. Over http://localhost that restriction does not apply.

Re-run after dropping in a new batch; labels already made are kept, since the
page stores them in the browser under the image's own path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic"}
SKIP = {"graded"}  # graded cards want the grade, not a defect label


def images() -> list[str]:
    found = []
    for path in sorted(HERE.rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        rel = path.relative_to(HERE)
        if rel.parts and rel.parts[0] in SKIP:
            continue
        found.append(str(rel))
    return found


def main() -> None:
    files = images()
    # Pairs found from the filename, so nothing has to be clicked: two shots
    # sharing a stem before _glare / _clean / _front / _back are the same
    # physical card. A pair holds border, foil and framing constant, so the
    # only difference left is what we are trying to detect.
    import re
    from pathlib import Path as _P

    pairs: dict[str, str] = {}
    for rel in files:
        stem = _P(rel).stem
        base = re.sub(r"[_-](glare|clean|front|back|a|b|1|2)$", "", stem,
                      flags=re.I)
        if base != stem:
            pairs[rel] = f"{_P(rel).parent}/{base}"

    import datetime

    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    html = (TEMPLATE.replace("__FILES__", json.dumps(files))
                    .replace("__PAIRS__", json.dumps(pairs))
                    .replace("__BUILT__", stamp))
    (HERE / "label.html").write_text(html)
    print(f"label.html ready — {len(files)} images")
    if "--serve" not in sys.argv:
        print("run with --serve to view it "
              "(opening the file directly shows blank images)")
        return

    import functools
    import http.server
    import socketserver
    import threading
    import webbrowser

    class NoCache(http.server.SimpleHTTPRequestHandler):
        """Browsers hold onto HTML hard, and http.server sends nothing to
        stop them. A cached page ignores every regeneration silently, which
        looks like the fix not working rather than never arriving."""

        def end_headers(self):
            self.send_header("Cache-Control",
                             "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def log_message(self, *args):
            pass

    handler = functools.partial(NoCache, directory=str(HERE))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", 8765), handler) as httpd:
        url = "http://localhost:8765/label.html"
        print(f"serving at {url}   (ctrl-c to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Card photo labeller</title>
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1d2026; --line:#2f343d;
          --ink:#e8eaed; --dim:#9aa3af; --on:#3b82f6; --ok:#16a34a; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui,-apple-system,sans-serif;
         background:var(--bg); color:var(--ink); display:flex; height:100vh; }
  #stage { flex:1; display:flex; align-items:center; justify-content:center;
           padding:16px; overflow:hidden; background:#0e1013; }
  #stage img { max-width:100%; max-height:100%; object-fit:contain;
               border-radius:6px; }
  #broken { color:#f87171; font-size:13px; text-align:center; max-width:520px;
            line-height:1.7; }
  #broken code { color:var(--dim); word-break:break-all; }
  #side { width:330px; flex:none; background:var(--panel); padding:16px;
          overflow-y:auto; border-left:1px solid var(--line); }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--dim); margin:18px 0 8px; font-weight:600; }
  h2:first-child { margin-top:0; }
  .row { display:flex; flex-wrap:wrap; gap:6px; }
  button.opt { background:#262b33; color:var(--ink); border:1px solid var(--line);
        border-radius:6px; padding:7px 11px; cursor:pointer; font-size:13px; }
  button.opt:hover { border-color:#4b5563; }
  button.opt[aria-pressed="true"] { background:var(--on); border-color:var(--on);
        color:#fff; }
  #corners { display:grid; grid-template-columns:repeat(3,1fr); gap:5px;
             width:170px; }
  #corners button { aspect-ratio:1; padding:0; font-size:13px;
                    font-variant-numeric:tabular-nums; }
  #corners .spacer { visibility:hidden; }
  textarea { width:100%; background:#262b33; color:var(--ink); border:1px solid
        var(--line); border-radius:6px; padding:8px; font:inherit; resize:vertical; }
  #nav { display:flex; gap:8px; align-items:center; margin-top:18px;
         padding-top:14px; border-top:1px solid var(--line); }
  #nav button { flex:1; padding:10px; font-weight:600; background:var(--on);
        color:#fff; border:none; border-radius:6px; cursor:pointer; }
  #nav button.ghost { background:#262b33; color:var(--ink); flex:0 0 44px;
        border:1px solid var(--line); }
  #count { color:var(--dim); font-size:12px; margin-top:10px; }
  #name { color:var(--dim); font-size:11px; word-break:break-all; margin-top:4px; }
  #built { color:#4b5563; font-size:10px; margin-top:6px; }
  #done { width:100%; margin-top:12px; padding:11px; background:var(--ok);
        color:#fff; border:none; border-radius:6px; font-weight:600;
        cursor:pointer; font-size:14px; }
  kbd { background:#262b33; border:1px solid var(--line); border-radius:3px;
        padding:0 4px; font-size:11px; color:var(--dim); }
  #hint { margin-top:12px; font-size:12px; color:var(--dim); line-height:1.7; }
  .sub { font-size:12px; color:var(--dim); margin-top:6px; }
</style></head><body>

<div id="stage"><img id="shot" alt=""><div id="broken" hidden></div></div>

<div id="side">
  <h2>The card <kbd>0</kbd> = no defects</h2>
  <div class="row" id="has-card"></div>
  <h2>The photograph <kbd>.</kbd> = nothing wrong</h2>
  <div class="row" id="has-photo"></div>


  <h2>Where — numpad <kbd>1</kbd>–<kbd>9</kbd></h2>
  <div id="corners"></div>

  <div id="wide-wrap" hidden>
    <h2>Which border is WIDEST</h2>
    <div class="row" id="wide"></div>
    <div class="sub">the side with more border — mark both if both axes
      are off</div>
  </div>

  <h2>Holder — tick all that apply</h2>
  <div class="row" id="holder"></div>

  <h2>Face</h2>
  <div class="row" id="face"></div>

  <h2>Same card as… <kbd>p</kbd></h2>
  <div class="row">
    <button class="opt" id="pair" type="button">same card as previous</button>
  </div>
  <div id="pairnote"></div>

  <h2>Notes</h2>
  <textarea id="notes" rows="3" placeholder="anything that matters"></textarea>

  <div id="nav">
    <button class="ghost" id="prev">←</button>
    <button id="next">Next <kbd>↵</kbd></button>
  </div>
  <div id="count"></div>
  <div id="name"></div>
  <div id="built">built __BUILT__</div>
  <button id="done">Download labels.csv</button>
  <button id="load" class="opt" style="width:100%;margin-top:8px">
    Load labels.csv from disk</button>
  <button id="reset" class="opt" style="width:100%;margin-top:8px">
    Reset all labels</button>
  <div id="hint">
    Unsure? Leave it blank. A wrong label becomes fake ground truth, which is
    worse than a missing one.<br>
    Progress is saved in this browser as you go.<br>
    Shot the same card twice? Put the two next to each other and press
    <kbd>p</kbd> on the second — a pair is worth several loose photos.
  </div>
</div>

<script>
const FILES = __FILES__;
const PAIRS = __PAIRS__;   // filename-derived, so pairs need no clicking
// The engine's own taxonomy, so a label maps onto a defect type rather than
// onto a word I invented. Grouped by category, plus the photo-level
// conditions that are properties of the PICTURE rather than the card.
const HAS = [
  {g:"card", v:"corners:whitening"},  {g:"card", v:"corners:rounding"},
  {g:"card", v:"corners:fraying"},    {g:"card", v:"edges:whitening"},
  {g:"card", v:"edges:chipping"},     {g:"card", v:"edges:roughness"},
  {g:"card", v:"surface:scratches"},  {g:"card", v:"surface:print_lines"},
  {g:"card", v:"surface:dimples"},    {g:"card", v:"surface:stains"},
  {g:"card", v:"surface:gloss_break"},{g:"card", v:"surface:crease"},
  {g:"card", v:"surface:paper_loss"},
  {g:"card", v:"centering:off_center"}, {g:"card", v:"centering:miscut"},
  {g:"photo", v:"glare"},             {g:"photo", v:"blur"},
  {g:"photo", v:"underexposed"},      {g:"photo", v:"occluded"},
  {g:"photo", v:"photo_ok"},
  {g:"card", v:"clean"},
];
// Nine, matching the engine's own vocabulary once edges were corrected to
// side names. Laid out as a numpad so the key you press is where you point.
const REGIONS = ["top_left","top","top_right",
                 "left","center","right",
                 "bottom_left","bottom","bottom_right"];
// How much plastic sits between the camera and the card, and how reflective
// it is. Not cosmetic: a one-touch is thick acrylic and throws far more
// glare than a toploader, which changes what the imaging layer sees.
const HOLDER = ["sleeve","toploader","one_touch","card_saver","screwdown",
                "slab","nothing"];
const FACE = ["front","back"];
const KEY_REGION = {"7":"top_left", "8":"top",    "9":"top_right",
                    "4":"left",     "5":"center", "6":"right",
                    "1":"bottom_left","2":"bottom","3":"bottom_right"};
const WIDE = ["left","right","top","bottom"];
const HELP = {
  "corners:whitening": "corner colour lifting to white — fibre showing through",
  "corners:rounding": "corner no longer a sharp point; blunted or soft",
  "corners:fraying": "corner layers separating, fuzzy or feathered",
  "edges:whitening": "white showing along an edge where colour has worn",
  "edges:chipping": "small pieces missing from the edge; nicks",
  "edges:roughness": "edge ragged or rough rather than cleanly cut",
  "surface:scratches": "thin LINES scored into the surface (an area of "
                     + "missing shine is gloss_break, not this)",
  "surface:print_lines": "line from the printing process, often dead straight",
  "surface:dimples": "small dents or indentations",
  "surface:stains": "discolouration, marks, residue",
  "surface:gloss_break": "a DULL PATCH where the shine is missing. An area, "
                       + "not a line — the surface need not be gouged. Often "
                       + "only visible when the card is TILTED under a light; "
                       + "if you can see it in hand but not in the photo, say "
                       + "so in the notes.",
  "surface:crease": "a fold or bend line through the card stock",
  "surface:paper_loss": "the surface layer actually TORN or lifted away — "
                      + "more than lost shine",
  "centering:off_center": "cut correctly, PRINTED off-centre — one border "
                        + "wider than the opposite one (60/40, 70/30). "
                        + "Tick it and a 'which border is widest' control "
                        + "appears.",
  "centering:miscut": "the CUT went wrong — a sliver of the neighbouring "
                    + "card visible, or a border missing entirely on one "
                    + "side. Rarer and usually more severe.",
  "clean": "this CARD has no defects (says nothing about the photograph)",
  glare: "a bright reflection washing out part of the picture",
  blur: "out of focus, or camera shake",
  underexposed: "too dark to see detail",
  occluded: "something BLOCKING the card — a thumb, a sticker, the "
          + "holder's clip, another card overlapping",
  photo_ok: "this PHOTOGRAPH has no problems (says nothing about the card)",
  sleeve: "thin soft plastic — a penny sleeve",
  toploader: "thin rigid PVC sleeve, open at the top",
  one_touch: "magnetic case: two thick acrylic halves held shut by magnets "
           + "(Ultra Pro One-Touch and similar). Very reflective.",
  card_saver: "semi-rigid holder, bends a little — the kind used to submit "
            + "cards for grading",
  screwdown: "rigid case held together by screws at the corners",
  slab: "sealed graded case with a label — PSA, BGS, SGC",
  nothing: "bare card, no holder",
};
const STORE = "card-labels-v1";

let i = 0;
let data = {};
try { data = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}

// The record shape has grown while labelling was already under way, so a
// stored record can predate a field the current code expects. Normalising
// only the record on screen left the others malformed, and anything that
// walks ALL of them — the "labelled" count in draw() — then threw on every
// redraw, which looks exactly like the Next button having stopped working.
// A FUNCTION, not a shared object. `{...SHAPE}` is a shallow copy, so every
// record ended up sharing the same has/regions/holder/wide arrays — and
// pushing to one pushed to all of them. Every card showed the same labels.
const blank = () => ({has: [], regions: [], wide: [], holder: [],
                      face: "", notes: "", card: ""});
function normalise(r) {
  const out = {...blank(), ...(r || {})};
  for (const k of ["has","regions","wide","holder"]) {
    if (typeof out[k] === "string") out[k] = out[k] ? [out[k]] : [];
    if (!Array.isArray(out[k])) out[k] = [];
  }
  for (const k of ["face","notes","card"]) {
    if (typeof out[k] !== "string") out[k] = "";
  }
  return out;
}
for (const k of Object.keys(data)) data[k] = normalise(data[k]);

const rec = () => {
  const r = (data[FILES[i]] ||= blank());
  if (typeof r.holder === "string") r.holder = r.holder ? [r.holder] : [];
  if (!r.card && PAIRS[FILES[i]]) r.card = PAIRS[FILES[i]];
  return r;
};

// A pair is worth several unpaired shots: same card, same border, same foil,
// so the ONLY difference is what we are trying to detect. Without it a
// threshold can end up separating chrome from paper rather than glare from
// clean, which is exactly how the corner metric ended up measuring border
// brightness instead of damage.
function pairWithPrevious() {
  if (i === 0) return;
  const prev = data[FILES[i - 1]];
  if (!prev) return;
  const id = prev.card || (prev.card = "card-" + String(i).padStart(3, "0"));
  rec().card = rec().card === id ? "" : id;
  save();
}
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(data)); } catch(e){} };

function chips(host, values, key, multi) {
  host.innerHTML = "";
  values.forEach(v => {
    const b = document.createElement("button");
    b.className = "opt"; b.type = "button";
    b.textContent = v.includes(":") ? v.split(":")[1].replace(/_/g, " ")
                                    : v.replace(/_/g, " ");
    b.title = HELP[v] || v;
    b.dataset.v = v;
    b.onclick = () => { toggle(key, v, multi); draw(); };
    host.appendChild(b);
  });
}
function toggle(key, value, multi) {
  const r = rec();
  if (multi) {
    const at = r[key].indexOf(value);
    at === -1 ? r[key].push(value) : r[key].splice(at, 1);
    if (key === "has") {
      // Each group carries its own "nothing wrong", and each is exclusive
      // ONLY within its group. A clean card in a bad photograph is the
      // commonest case here — it is the clean half of every glare pair — so
      // saying the card is fine must not say anything about the picture.
      const NONE = {card: "clean", photo: "photo_ok"};
      const group = (HAS.find(h => h.v === value) || {}).g;
      const none = NONE[group];
      if (!none) return save();
      const others = HAS.filter(h => h.g === group && h.v !== none).map(h => h.v);
      if (value === none && at === -1) {
        r.has = r.has.filter(x => !others.includes(x));
      } else if (others.includes(value) && at === -1) {
        r.has = r.has.filter(x => x !== none);
      }
    }
  } else {
    r[key] = r[key] === value ? "" : value;
  }
  save();
}
function mark(host, key, multi) {
  [...host.children].forEach(b => {
    const v = b.dataset.v;
    const on = multi ? rec()[key].includes(v) : rec()[key] === v;
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

const cornerHost = document.getElementById("corners");
const NUMPAD = {top_left:"7", top:"8", top_right:"9",
                left:"4", center:"5", right:"6",
                bottom_left:"1", bottom:"2", bottom_right:"3"};
function buildCorners() {
  cornerHost.innerHTML = "";
  REGIONS.forEach(v => {
    const b = document.createElement("button");
    b.className = "opt"; b.type = "button";
    b.textContent = NUMPAD[v];
    b.title = v.replace(/_/g, " ");
    b.dataset.v = v;
    b.onclick = () => { toggle("regions", v, true); draw(); };
    cornerHost.appendChild(b);
  });
}

function draw() {
  try { render(); } catch (err) {
    const broken = document.getElementById("broken");
    document.getElementById("shot").hidden = true;
    broken.hidden = false;
    broken.innerHTML = "something went wrong drawing this card<br><code>" +
      String(err) + "</code><br><br>Next and Prev still work. " +
      "If it persists, click Reset below and reload.";
  }
}

function render() {
  const f = FILES[i];
  const img = document.getElementById("shot");
  const broken = document.getElementById("broken");
  // A blank stage is indistinguishable from a bug. Say which path failed —
  // the usual cause is a stale tab still pointing at files that have since
  // been moved or renamed.
  img.onerror = () => {
    img.hidden = true; broken.hidden = false;
    broken.innerHTML = "could not load<br><code>" + f + "</code><br><br>" +
      "If the files have moved, re-run <code>make_labeller.py</code> " +
      "and reload this page.";
  };
  img.onload = () => { img.hidden = false; broken.hidden = true; };
  img.src = encodeURI(f);
  document.getElementById("count").textContent =
    `${i + 1} of ${FILES.length}  ·  ${Object.values(data).filter(d => (d.has || []).length).length} labelled`;
  document.getElementById("name").textContent = f;
  document.getElementById("notes").value = rec().notes;
  const mine = rec().card;
  document.getElementById("pair").setAttribute("aria-pressed", mine ? "true" : "false");
  document.getElementById("pairnote").textContent = mine
    ? `linked as ${mine}` : "";
  ["has-card","has-photo"].forEach(
    id => mark(document.getElementById(id), "has", true));
  mark(document.getElementById("holder"), "holder", true);
  // Only asked when it means something. Overloading the region grid with
  // "where the defect is" AND "which border is widest" made one control
  // answer two different questions.
  const offCentre = rec().has.includes("centering:off_center");
  document.getElementById("wide-wrap").hidden = !offCentre;
  mark(document.getElementById("wide"), "wide", true);
  mark(document.getElementById("face"), "face", false);

  // Say when there is nowhere further to go. A live-looking button that
  // does nothing reads as broken, and sent someone hunting for a bug that
  // was not there.
  const atEnd = i === FILES.length - 1, atStart = i === 0;
  const next = document.getElementById("next");
  next.disabled = atEnd;
  next.textContent = atEnd ? "last card — all done" : "Next";
  next.style.opacity = atEnd ? "0.45" : "1";
  next.style.cursor = atEnd ? "default" : "pointer";
  document.getElementById("prev").disabled = atStart;
  document.getElementById("prev").style.opacity = atStart ? "0.45" : "1";
  [...cornerHost.children].forEach(b => {
    if (b.dataset.v) b.setAttribute("aria-pressed",
      rec().regions.includes(b.dataset.v) ? "true" : "false");
  });
}

const byGroup = g => HAS.filter(h => h.g === g).map(h => h.v);
chips(document.getElementById("has-card"), byGroup("card"), "has", true);
chips(document.getElementById("has-photo"), byGroup("photo"), "has", true);

chips(document.getElementById("holder"), HOLDER, "holder", true);
chips(document.getElementById("wide"), WIDE, "wide", true);
chips(document.getElementById("face"), FACE, "face", false);
buildCorners();

const go = d => { i = Math.max(0, Math.min(FILES.length - 1, i + d)); draw(); };
document.getElementById("next").onclick = () => go(1);
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("notes").oninput = e => { rec().notes = e.target.value; save(); };
document.getElementById("pair").onclick = () => { pairWithPrevious(); draw(); };

document.onkeydown = e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "Enter" || e.key === "ArrowRight") { go(1); return; }
  if (e.key === "ArrowLeft") { go(-1); return; }
  if (e.key === "0") { toggle("has", "clean", true); draw(); return; }
  if (e.key === ".") { toggle("has", "photo_ok", true); draw(); return; }
  if (e.key === "p") { pairWithPrevious(); draw(); return; }
  const r = KEY_REGION[e.key];
  if (r) { toggle("regions", r, true); draw(); }
};

document.getElementById("done").onclick = () => {
  const esc = s => `"${String(s).replace(/"/g, '""')}"`;
  // Provenance comes from the path, not from you typing it. A screenshot is
  // a downscaled, resampled, colour-managed view of the seller's image, so
  // mixing the two lets a threshold fit PROVENANCE instead of damage.
  const source = f => f.startsWith("screenshots/") ? "screenshot"
                    : f.startsWith("listings/") ? "listing_image" : "";
  const rows = [["file","source","card","has","regions","wide_border",
                 "holder","face","notes"].join(",")];
  FILES.forEach(f => {
    const d = data[f];
    if (!d || (!d.has.length && !d.holder.length && !d.face && !d.notes)) return;
    rows.push([f, source(f), d.card || "", d.has.join("|"), d.regions.join("|"),
               (d.wide || []).join("|"), (d.holder || []).join("|"),
               d.face, d.notes]
      .map(esc).join(","));
  });
  const blob = new Blob([rows.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.csv";
  a.click();
};

// Labels live in the browser, which makes them one cleared cache — or one
// bug in this file — away from gone. Reading the exported CSV back turns
// that into a recoverable situation.
document.getElementById("load").onclick = async () => {
  let text;
  try {
    const res = await fetch("labels.csv", {cache: "no-store"});
    if (!res.ok) throw new Error(res.status);
    text = await res.text();
  } catch (e) {
    alert("no labels.csv in this folder yet"); return;
  }
  const lines = text.trim().split("\n");
  const head = split(lines.shift());
  let loaded = 0;
  for (const line of lines) {
    const cells = split(line);
    const row = Object.fromEntries(head.map((h, n) => [h, cells[n] ?? ""]));
    if (!row.file) continue;
    data[row.file] = normalise({
      has: (row.has || "").split("|").filter(Boolean),
      regions: (row.regions || "").split("|").filter(Boolean),
      wide: (row.wide_border || "").split("|").filter(Boolean),
      holder: (row.holder || "").split("|").filter(Boolean),
      face: row.face || "", notes: row.notes || "", card: row.card || "",
    });
    loaded++;
  }
  save(); draw();
  alert(`loaded ${loaded} labels`);
};

function split(line) {
  const out = []; let cur = "", quoted = false;
  for (let n = 0; n < line.length; n++) {
    const ch = line[n];
    if (quoted) {
      if (ch === '"' && line[n + 1] === '"') { cur += '"'; n++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

document.getElementById("reset").onclick = () => {
  if (!confirm("Delete every label stored in this browser? "
             + "Download the CSV first if you want to keep them.")) return;
  data = {}; save(); draw();
};

draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
