# Nothing to do here yet

This folder is a placeholder for a FUTURE dataset. It is not part of the
current ask, and it is NOT for photos of slabs.

## What does NOT go here

A photograph of a graded card in its case. The engine screens RAW cards —
that is the whole premise, "does this raw card have a chance at a 10" — and a
slab is a different object: recessed behind thick plastic, label covering the
frame, heavy reflection off the case. Boundary detection finds the slab, and
cropping the slab away does not help (tested: the same photo was rejected at
15%, 8%, 3% and 0% margin, because cropping pushes the card toward filling
the frame, which is the direction that fails).

A PSA 10 is also only ever a "clean" example, so it cannot calibrate a
detector that needs damaged ones. Lower grades would be more useful than 10s.

## What DOES go here, eventually

    the photo of the card while it was still RAW
      + the grade it came back with

That pairing is ground truth for the whole verdict rather than for one
detector. It answers the question that actually costs money: would the engine
have REJECTED something that graded 10? It is what the append-only prediction
history and the "compare CV vs Claude vs combined vs actual outcome"
requirement exist for, and nothing else substitutes for it.

The catch is that it requires photographing the card BEFORE sending it in. A
photo taken after it comes back tells us the grade but not what the engine
would have seen.

## So

Going forward, photograph cards front and back before you submit them,
ordinary phone shots, and note what comes back. Drop them here as:

    <cert-or-any-id>_front.jpg
    <cert-or-any-id>_back.jpg
    outcomes.csv       id,grade,grader,notes

Even twenty of these eventually is worth more than any number of synthetic
fixtures. But gather them on their own track — nothing in the current work
waits on it.
