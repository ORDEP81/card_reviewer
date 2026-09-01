"""Card review engine (subsystem A).

Importing this package must stay cheap: `cv2` and `anthropic` are imported
lazily inside the functions that need them, the same discipline
`card_reviewer.knowledge` follows. NumPy is imported at module scope in the
`imaging` package only — nothing in `enums`, `policies` or `storage` may
import `imaging`, which is what keeps the policy layer light.
"""
