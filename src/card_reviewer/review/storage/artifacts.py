"""Content-addressed storage for images and derived artifacts.

SQLite holds records; the filesystem holds bytes. Large blobs never go into
the database. Originals are preserved untouched (non-negotiable rule 6).

Artifact ids are derived from *content and logical location* — never from an
absolute path. Cached `EvidenceRef`s outlive the process that wrote them, so
an id that encoded where the store happened to sit would dangle the moment
the data directory moved or a new process opened it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Separates the artifact id from the human-readable name on disk, so a
#: directory listing stays legible while lookup stays exact.
_SEP = "__"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._images = self.root / "images"
        self._crops = self.root / "crops"
        self._images.mkdir(parents=True, exist_ok=True)
        self._crops.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._reindex()

    # -- ids ------------------------------------------------------------

    @staticmethod
    def derived_id(image_hash: str, kind: str, name: str, data: bytes) -> str:
        """Deterministic across processes, machines and store locations.

        Content is part of the id, so two stages writing the same logical
        name with different bytes get different ids instead of one silently
        overwriting the other.
        """
        seed = f"{image_hash}|{kind}|{name}|{_sha(data)}"
        return _sha(seed.encode("utf-8"))[:32]

    # -- writing --------------------------------------------------------

    def put_image(self, data: bytes) -> str:
        image_hash = _sha(data)
        dest = self._images / image_hash
        if not dest.exists():
            dest.write_bytes(data)
        self._index[image_hash] = dest
        return image_hash

    def put_derived(self, image_hash: str, kind: str, name: str,
                    data: bytes) -> str:
        artifact_id = self.derived_id(image_hash, kind, name, data)
        directory = self._crops / image_hash / kind
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / f"{artifact_id}{_SEP}{name}"
        if not dest.exists():
            dest.write_bytes(data)
        self._index[artifact_id] = dest
        return artifact_id

    # -- reading --------------------------------------------------------

    def path_of(self, artifact_id: str) -> Path:
        if artifact_id not in self._index:
            # A store opened over existing data has not seen these writes.
            self._reindex()
        try:
            return self._index[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact {artifact_id!r}") from exc

    def read(self, artifact_id: str) -> bytes:
        return self.path_of(artifact_id).read_bytes()

    # -- index ----------------------------------------------------------

    def _reindex(self) -> None:
        """Rebuild id -> path from disk, so a restart resolves what a previous
        process wrote."""
        for path in self._images.iterdir():
            if path.is_file():
                self._index[path.name] = path
        for path in self._crops.rglob(f"*{_SEP}*"):
            if path.is_file():
                self._index[path.name.split(_SEP, 1)[0]] = path
