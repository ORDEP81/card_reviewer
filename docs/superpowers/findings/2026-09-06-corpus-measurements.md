# What the 120-photo corpus says (2026-09-06)

Measured with `.dev/calibrate.py` and `.dev/separate.py` at commit `a827d6b`.
Populations selected by the `clean` label, not by absence of other labels —
see `tests/review/test_label_vocabulary.py` for why that distinction changed
the numbers.

## Framing

    117 raw cards (2 slabs excluded)
     88 reach the measurement stages
     29 declined by geometry
      0 questionable quads

## Verdicts, single front photograph, OFF mode

    clean-labelled     REVIEW 31   INSUFFICIENT_IMAGES 21   REJECT 1
    defect-labelled    REVIEW 41   INSUFFICIENT_IMAGES 22   REJECT 0

No PASS is possible here: every candidate is front-only, and a missing back
prevents PASS by policy. REVIEW is the correct ceiling.

The single REJECT is `listings/glare/card03_clean.webp`, on vertical
centering 58/42. Inspected: the detected quad bounds the card rather than
the sleeve, and the top border is genuinely thicker than the bottom. The
call is defensible; the card was labelled "clean" in the sense of no glare
and no wear, because it came from the glare batch.

## The rank score is currently measuring the corner detector's noise

Clean-labelled cards, bucketed by how many corner anomalies fired:

    0 anomalies   n= 9   median rank score 90
    1 anomaly     n= 3   median rank score 60
    2 anomalies   n= 6   median rank score 30
    3 anomalies   n= 8   median rank score 15
    4 anomalies   n=11   median rank score  0

The detector fires on 28 of 37 clean cards, and the score is very nearly a
linear function of that false-positive count. Clean and defect-labelled
cards therefore share a median rank score of 15: the ordering the score
exists to provide does not exist yet.

This does NOT breach I1. False anomaly candidates are not promoted to
confirmed defects and do not manufacture REJECTs — one reject in 53 clean
cards, and that one is real. The failure is in ranking, not in safety.

## Detector separation, unchanged

    glare      2/17 clear the clean extreme  (12% recall @ 0% FP)
    corners    0/16                          ( 0% recall @ 0% FP)
    surface    0/19                          ( 0% recall @ 0% FP)

No threshold on any measure tried divides the labelled classes.
