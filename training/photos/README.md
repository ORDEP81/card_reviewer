# Training photos

120 labelled photographs of real trading cards, and the labels for them.

This is the ACCEPTANCE CORPUS for the imaging layer. It is committed rather
than kept on one machine because it is the only evidence that any of this
works on real input, and because it is not reproducible — these are one
person's cards and one person's judgement about what is wrong with them.

It has already earned that. Twenty-eight of these photographs showed that
geometry rejected 24 of 26 real listing images, a defect four independent
reviews, 1168 tests and an entire synthetic corpus had missed: the fixtures
assumed the card fills 59% of the frame, and real listings put it at
94-99.8%. The remaining 92 then showed that none of the three CV defect
detectors separates the classes a human labelled.

## Layout

    screenshots/     the first 28, captured from a browser. Lower fidelity —
                     a screenshot is a downscaled, resampled, colour-managed
                     view of the seller's image — but they found the framing
                     defect and are kept for that.
    listings/        90+, saved with "Save image as" or shot directly.
      glare/         paired: same card, flash and diffuse
      clean/         no visible defects — the negative class
      corner_wear/
      miscut/        off-centre and true miscuts
      surface/       creases, print lines, scratches, dimples
    graded/          empty. See its README: NOT photos of slabs.

    labels.csv       what each photograph shows. THE most important file
                     here — without it the images are just pixels.

## labels.csv

    file          path, relative to this folder
    source        screenshot | listing_image, derived from the path. A
                  screenshot and a saved image differ in resolution,
                  resampling and colour, so a threshold must be checked
                  against this or it can end up separating PROVENANCE.
    card          shared id linking two shots of the same physical card
    has           defects, in the engine's own taxonomy (corners:rounding,
                  surface:crease, ...), plus photograph conditions (glare,
                  blur, underexposed, occluded) and the two explicit
                  "nothing wrong" markers, clean and photo_ok
    regions       top_left / top / ... / center. Shared across a card's
                  defects, so it cannot say WHICH defect is where when a
                  card has more than one.
    wide_border   for off-centre: which border is widest
    holder        sleeve, toploader, one_touch, card_saver, screwdown,
                  slab, nothing. Multiple, since a card can be in a sleeve
                  inside a toploader.
    face          front | back
    notes         free text

A blank field means UNSURE, deliberately. It is not the same as "fine" —
that is what `clean` and `photo_ok` are for. Anything blank should be
treated as unknown rather than negative.

## Using it

    uv run python .dev/calibrate.py    # what reaches measurement, and why
    uv run python .dev/separate.py     # does a measure separate the labels

## Labelling more

    python3 training/photos/make_labeller.py --serve

Serve it; do not open label.html directly. Browsers block a file:// page from
reading local images, which looks exactly like a broken page. label.html is
gitignored — it embeds a file list and is regenerated.
