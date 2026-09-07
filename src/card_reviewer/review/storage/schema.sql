-- Card review engine schema v1.
--
-- Append-only by discipline: stage_result and review rows are never updated,
-- so a later analyzer improvement can be compared against what the previous
-- one concluded (spec §4, "history is never overwritten").

CREATE TABLE IF NOT EXISTS candidate (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    listing_url TEXT,
    listing_id TEXT,
    title TEXT,
    -- Listing provenance only. Never read by the grading path; the adapter
    -- drops it when building ResolvedCandidate (non-negotiable rule 14).
    asking_price TEXT,
    supplied_card_type TEXT,
    supplied_set TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image (
    image_hash TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    format TEXT,
    bytes INTEGER,
    created_at TEXT
);

-- Many-to-many: the same photograph across two listings is stored and
-- analyzed once, and linked twice.
CREATE TABLE IF NOT EXISTS candidate_image (
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    image_hash TEXT NOT NULL REFERENCES image(image_hash),
    supplied_role TEXT,
    source_url TEXT,
    ordering INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (candidate_id, image_hash)
);

-- Validated successes ONLY. A row here means the stage ran to completion and
-- passed schema validation.
CREATE TABLE IF NOT EXISTS stage_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    producer_signature TEXT NOT NULL,
    output_json TEXT NOT NULL,
    versions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    image_hash TEXT,
    candidate_id TEXT,
    UNIQUE (stage, input_fingerprint, producer_signature)
);

-- Failures, timeouts, provider errors, malformed responses. Diagnostics and
-- cost accounting only; NEVER satisfies a cache lookup.
CREATE TABLE IF NOT EXISTS stage_attempt (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    input_fingerprint TEXT,
    producer_signature TEXT,
    error_kind TEXT NOT NULL,
    error_detail TEXT,
    cost_usd REAL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL,
    image_hash TEXT,
    candidate_id TEXT
);

CREATE TABLE IF NOT EXISTS routing_decision (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    policy_version TEXT NOT NULL,
    mode TEXT NOT NULL,
    call_vision INTEGER NOT NULL,
    trigger_reasons TEXT,
    input_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    mode TEXT NOT NULL,
    routing_decision_id INTEGER NOT NULL REFERENCES routing_decision(id),
    verdict TEXT NOT NULL,
    psa10_candidate TEXT NOT NULL,
    psa10_rank_score INTEGER,
    rankable INTEGER NOT NULL,
    estimated_psa_grade TEXT,
    review_confidence TEXT NOT NULL,
    coverage TEXT NOT NULL,
    heuristic_result_id INTEGER REFERENCES stage_result(id),
    coverage_provisional_result_id INTEGER REFERENCES stage_result(id),
    coverage_result_id INTEGER REFERENCES stage_result(id),
    combine_result_id INTEGER REFERENCES stage_result(id),
    -- Nullable: this is how OFF mode is represented, rather than as a
    -- missing stage (spec §5).
    vision_result_id INTEGER REFERENCES stage_result(id),
    rubric_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- No price. No purchase flag. Build plan §26's calibration record asks for
-- grades, not cost; storing price beside returned grades puts ROI analysis
-- one join away, which rule 14 forbids in this repository.
CREATE TABLE IF NOT EXISTS candidate_outcome (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    status TEXT NOT NULL,
    occurred_on TEXT,
    notes TEXT
);

-- Multiple rows per candidate are expected: cards get returned ungraded,
-- resubmitted, cracked and resubmitted, or crossed to another grader.
CREATE TABLE IF NOT EXISTS grading_submission (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    grader TEXT NOT NULL,
    submitted_on TEXT,
    service_tier TEXT,
    returned_on TEXT,
    grade TEXT,
    cert_number TEXT,
    status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stage_result_lookup
    ON stage_result(stage, input_fingerprint, producer_signature);
CREATE INDEX IF NOT EXISTS idx_review_candidate ON review(candidate_id);
