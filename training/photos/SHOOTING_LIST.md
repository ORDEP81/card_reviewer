# What to shoot

## Quick reference

| # | group | how many | folder | name them |
|---|---|---|---|---|
| 1 | Glare — same card, 2 lightings | 10 cards = 20 shots | `listings/glare/` | `card01_glare.jpg` + `card01_clean.jpg` |
| 2 | Corner wear | ~15 | `listings/corner_wear/` | anything, e.g. `wear01.jpg` |
| 3 | Off-centre 70/30+, corners clean | ~10 | `listings/miscut/` | anything, e.g. `offcentre01.jpg` |
| 4 | Surface: crease, print line, scratch, gloss break, dimple, stain | ~12 | `listings/surface/` | anything, e.g. `crease01.jpg` |
| 5 | Clean, no defects | ~5 for VARIETY | `listings/clean/` | anything, e.g. `clean01.jpg` |
| 6 | Backs — only if free, 5-8 | pairs only | same folder as its front | `card01_front.jpg` + `card01_back.jpg` |

Names only matter for PAIRS. Two files sharing a stem before `_glare`/`_clean`
or `_front`/`_back` are treated as the same physical card automatically.
Everything else can be called anything, but keep to letters, numbers, dashes
and underscores — macOS screenshot names contain a narrow no-break space that
breaks in browsers.

## Where the photos come from

Either your own cards or marketplace listings. Both are wanted, and the glare
pairs can ONLY be your own — you cannot ask a seller to re-shoot the same
card under different lighting.

    your own cards      glare pairs (the only source), corner wear,
                        off-centre. You know the ground truth because you
                        can hold the card.
    listings            the genuine input distribution: other people's
                        framing, holders, backgrounds. Save the image
                        (right-click -> "Save image as"), never a
                        screenshot. No pairs possible.

Shooting your own: a phone, hand-held, whatever light is in the room, the
card in whatever holder it lives in. NO tripod, NO lightbox, NO careful even
lighting, NO cropping.

That restraint is the point. If every self-shot card is well lit, square-on
and neatly framed, the thresholds get calibrated on a distribution that does
not occur — which is exactly the mistake that made the whole imaging layer
reject 24 of the first 26 real photographs. Shoot them carelessly, the way a
seller in a hurry would. Bad photos are data.

Phone settings: turn off auto-enhance/HDR if that is easy, and make sure you
are getting JPEG rather than HEIC — iPhones default to HEIC and OpenCV cannot
read it.

Whatever the source: do NOT crop, rotate, straighten or enhance, and do not
convert the file type. Re-saving is a second lossy generation, and those
artefacts land in the high-frequency detail the surface and sharpness
measures read.

---

## 1. Glare — 10 cards, 2 shots each        -> listings/glare/

The only category shot twice, because glare is the only condition you can
turn on and off on the same card. Same card, same framing, two lightings:

    flash or raking light   -> glare
    diffuse / window light  -> clean

Name the two with a matching stem and they pair themselves:

    card01_glare.jpg   card01_clean.jpg
    card02_glare.jpg   card02_clean.jpg

At least 3-4 should be LIGHT-BORDERED PAPER cards, not chrome or foil. The
open bug is that a wide white border and a blown-out corner look identical;
foil glare is a different problem and the 28 screenshots already cover it.

Mark WHICH corners are glared when labelling — that is the part I most need.

## 2. Corner wear — ~15 photos              -> listings/corner_wear/

Single shots, no pairing.

Best case is one card with SOME corners worn and some sharp: same border,
same lighting, only the wear differs. That is a stronger control than two
separate cards, and it is one photo instead of two.

Label which corners are worn and which are clean.

## 3. Off-centre — ~10 photos                -> listings/miscut/

Single shots. 70/30 or worse.

OFF-CENTRE and MISCUT are different defects and the labeller now separates
them. Off-centre means the card was cut correctly but PRINTED off-centre, so
one border is wider than the opposite one — that is what a 70/30 ratio
describes, and it is what the engine measures. A miscut means the CUT went
wrong: a sliver of the neighbouring card is visible, or a border is missing
entirely. Collectors say "miscut" for both.

Off-centre is what this batch is for. True miscuts are welcome too — label
them as such, since a neighbouring card inside the frame confounds the border
measurement in its own way.

### Say WHICH WAY it is off-centre

Tick `off_center` and a "which border is WIDEST" control appears — left,
right, top, bottom, and both if both axes are off. It has its own control
rather than reusing the region grid, because the grid answers "where is the
defect" and this answers a different question.

This matters more than it sounds. The engine measures horizontal and vertical
centering as separate axes, and the vertical one was silently broken for
eleven commits while the horizontal one worked — a tilted card read 50.7
across and 20.4 down against a true 50/50, and was REJECTED for a miscut it
did not have. Every test written at the time happened to exercise only the
horizontal axis. A label that names the axis is what catches that.

Corners must be CLEAN — the bug is that a miscut card raises FALSE corner and
edge damage, so a miscut card that also has real corner wear tells me
nothing. Label as miscut with no corner regions marked.

## 4. Surface — ~12 photos                   -> listings/surface/

The thinnest coverage in the corpus: 2 usable examples across SEVEN defect
types, and surface is the largest category in the taxonomy.

Spread across types rather than twelve scratches:

    crease         2-3   most severe, an automatic PSA-10 disqualifier,
                         and there are none at all in the corpus
    print_lines      2   usually dead straight, so most likely detectable
    scratches        3   commonest real defect; hard on foil
    gloss_break    2-3   subtle — the honest test of whether CV sees them
    dimples
    stains         1-2   rarer, but zero coverage
    paper_loss

Whole card, shot close. A card with a defect in one area and clean elsewhere
is the strongest single shot: same lighting, same finish, only the defect
differs.

These may overturn a conclusion already drawn. The surface detector is
currently SILENT by design — its thresholds sit above the clean maximum
because the synthetic populations overlapped completely. But that was
measured on synthetic "scratches" rendered as a bright noisy patch, which is
not what a real crease or print line looks like. These photos either overturn
that or confirm it properly.

They also decide whether OFF mode can ever assess surface at all. Today a
scratched card returns PASS in OFF mode, because nothing looks at the
surface.

## 5. Clean — about 5 more, chosen for VARIETY -> listings/clean/

No visible defects at all: sharp corners, no whitening or chipping, no
scratches or creases, reasonably centred. The general NEGATIVE CLASS — how
the code learns what a measurement reads when nothing is wrong, which is what
says where a threshold belongs.

14 are already usable, giving 56 clean corner readings, and the clean half of
each glare pair adds ten more. So COUNT is not the gap — VARIETY is. The 14
skew to sleeves and bare cards with only one one-touch. Pick for spread:

    one dark-bordered, one light/white-bordered
    one matte or paper stock (almost everything so far is chrome or foil)
    one in a one-touch (the holder that throws the most reflection)

Five chosen that way are worth more than fifteen more of the same.

Note the word "clean" is doing two jobs, which is my fault:

    card01_clean.jpg      half of a glare PAIR. The variable is the
                          LIGHTING, not the card. "This photo has no glare."
    listings/clean/       a card with no defects at all.

Do not agonise over whether a card is clean enough. Mostly clean with one
chipped corner? Keep it, label the corner. The LABEL is the truth and the
folder is a rough sort — a mislabelled card is expensive, a misfiled one
costs nothing.

---

## Also worth grabbing — but only 5-8, not for every card

BACKS. Only needed to exercise the coverage rules with real data (a missing
back prevents PASS; a usable front with no back stays PARTIAL and rankable).
Five to eight front/back pairs is plenty, and there is not one real back in
the corpus today.

They add nothing to the glare, corner-wear or miscut work — the defect is on
whichever face shows it.

A back ALWAYS needs its front. A back on its own cannot even be reviewed —
an unusable front means INSUFFICIENT_IMAGES and the card never gets that far.
Never file a back alone.

No constraints on which cards: damaged corners, any border colour, foil,
miscut all fine. These pairs exercise PLUMBING — did both faces arrive, did
role resolution assign them, does coverage behave with two faces instead of
one — and none of that cares what is on the card. The constraints
(light-bordered paper, clean corners) apply only to the glare and miscut
batches, where a confound would ruin the measurement.

So: grab a back when the listing already has one and it is a single extra
click. Do not go hunting. Name them `card01_front.jpg` / `card01_back.jpg`
and they pair automatically.

FRONT-ONLY listings are realistic and wanted too. Plenty of sellers post one
photo, and that is a production case the engine has to handle — not a gap in
the data.

## Not now

graded/ — see the README in that folder. Not photos of slabs.

---

## Holders

Tick every layer, not one: a card can be in a sleeve INSIDE a toploader.

    sleeve       thin soft plastic (penny sleeve)
    toploader    thin rigid PVC, open at the top
    one_touch    magnetic case — two thick acrylic halves held by magnets.
                 Very reflective, so it throws far more glare than a
                 toploader. Worth distinguishing for that reason.
    card_saver   semi-rigid, bends slightly; used for grading submissions
    screwdown    rigid case with screws at the corners
    slab         sealed graded case with a label (PSA, BGS, SGC)
    nothing      bare card

"Raw" is not a holder — it just means not slabbed, which is implied by not
ticking slab.

## Labelling — what label.html is for

It is a click-through tool for saying WHAT IS WRONG with each photo. Shooting
puts a card in a folder; labelling is what turns it into data — the folder is
a rough sort, the label is the truth.

Use it on EVERY photo, in every folder, including the 28 screenshots. A photo
with no label is a file nothing can learn from.

    python3 training/photos/make_labeller.py     # after adding photos
    open training/photos/label.html              # then RELOAD if already open

Big image on the left, buttons on the right. Per card, about five seconds:

    1-7          what is wrong (1 glare ... 7 clean). Several allowed.
    q w a s e    which region: top-left, top-right, bottom-left,
                 bottom-right, centre. THIS is the part most needed for
                 glare and corner wear.
    Enter        next card
    p            same card as the previous one (only if the filenames did
                 not already pair them)

Then "Download labels.csv" and leave the file in training/photos/.

Progress is saved in the browser as you go, so you can stop and come back.
If you moved or renamed files, re-run make_labeller.py and RELOAD the page —
a stale tab points at paths that no longer exist and shows a blank stage.

Leave anything you are unsure about BLANK. A guessed label becomes fake
ground truth and gets calibrated against. A missing one costs nothing.
