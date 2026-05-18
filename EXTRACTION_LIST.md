# Extraction List — Patterns Worth Harvesting

**Source:** Read-only evaluation of `agentmemory-01` (treated as a hostile/external repo).
**Mode:** Identify ideas/approaches/patterns. **Do NOT copy code.** Each entry below is to be re-implemented from scratch in vanilla code if/when we decide to harvest it.

Each entry follows the same shape:
- **Idea** — what's interesting (algorithmic or architectural)
- **Where it lives** — file-level pointer for future reference, NOT a copy target
- **Re-implementation notes** — how we'd write it ourselves, plus risks/caveats
- **Priority** — H / M / L (worth doing soon / worth doing eventually / nice-to-know)

---

## A. Retrieval & Search

### A1. Reciprocal Rank Fusion (RRF) with adaptive weight renormalization  [H]
- **Idea.** Combine 3 retrieval signals (lexical BM25, dense vector, knowledge-graph entity match) by converting each to a *rank* (not raw score) and summing `weight_i × 1/(K + rank_i)` with K≈60. The clever bit: if a signal returns no results (vector index empty, graph extraction failed), zero its weight and re-normalize the surviving weights to sum to 1, so the combined score doesn't collapse.
- **Where.** `src/state/hybrid-search.ts` — RRF + degradation logic.
- **Re-impl.** ~50 LOC pure function: takes three rank-lists, three weights, returns merged ordered list. Trivial unit tests. Keep K configurable.
- **Why it matters.** Robust to signal failure without per-modality calibration, dodges the "vector returns 0.001 but BM25 returns 12.4" normalization problem.

### A2. Triple-stream candidate retrieval with graph expansion from top-N vector hits  [M]
- **Idea.** Run BM25, vector, and graph entity search in parallel. Then take the **top-5 vector hits** and use them as seed nodes to expand the graph one hop further (`expandFromChunks(seeds, hops=1, fanout=5)`), *appending* those hops to the graph result set. The graph signal gets two sources: direct entity hits from query NER + neighbors of strong vector hits.
- **Where.** `src/state/hybrid-search.ts` `tripleStreamSearch`.
- **Dedup policy.** When the same `obsId` shows up in *both* graph sources, **min the rank and max the score** within the graph slot (NOT sum). Across the three signals (BM25 / vector / graph), each contributes its own independent rank into the RRF in A1 — never summed across signals either.
- **Re-impl.** Don't bother unless we have a graph. If we do: keep expansion conservative (1 hop, ≤5 neighbors) so it can't dominate.
- **Risk.** Hops compound — guardrails on both fan-out and recursion.

### A3. Reranker as a window-limited, lazy-loaded, fail-open afterburner  [H]
- **Idea.** After hybrid scoring + diversification, rerank only the **top 20** results with a quantized cross-encoder. The model is loaded lazily; concurrent callers `await` a single in-flight `loadPipeline()` promise (promise dedup). On *any* error (load, inference) it returns the un-reranked list unchanged. A static `pipelineUnavailable` flag short-circuits all future attempts once the model fails to load.
- **Where.** `src/state/reranker.ts` (~70 LOC, very compact).
- **Re-impl.** The architectural pattern is more valuable than the model choice. Always rerank a small head, fail open, cache the "broken" verdict.
- **Note.** Their model name (`ms-marco-MiniLM-L-6-v2`) is fine for English, but in our case pick freshly.

### A4. Session-aware result diversification  [M]
- **Idea.** Cap results per `sessionId` (max 3) in the top-K, then pad with leftovers if under-limit. Stops one verbose session from monopolizing search results.
- **Where.** `hybrid-search.ts` `diversifyBySession`.
- **Re-impl.** ~15 LOC. Useful for any retrieval where the source has a natural cluster identifier (session, document, author).

### A5. BM25 with prefix-expansion via sorted-term binary search  [M]
- **Idea.** Standard BM25 (k1=1.2, b=0.75) gets a cheap fuzz extension: maintain a lazily-built sorted array of all index terms. For each query term, `lowerBound` the sorted array and scan while the indexed term still has the query term as a prefix; score those hits with a **half-weight IDF (×0.5)**. No Levenshtein, no n-grams.
- **Where.** `src/state/search-index.ts`.
- **Re-impl.** Maintain sorted terms with a stale flag invalidated on `add`. Half-weight on prefix matches is the empirical knob.

### A6. Synonym expansion at search time, half-weight, stem-aware  [M]
- **Idea.** Hand-curated domain synonym groups (e.g., `[db, database, datastore]`). At search time, expand each query term with its synonyms at **weight 0.7**, deduped against terms already added. Synonyms and query both stemmed first so morphological variants don't get double-counted.
- **Where.** `src/state/synonyms.ts` + `search-index.ts`.
- **Re-impl.** Static map keyed by stems, build once at startup. We can crowdsource a small list for our domain — it's far cheaper than WordNet and far better than nothing.

### A7. Brute-force vector index with online top-K (no HNSW/IVF)  [L]
- **Idea.** For "small-corpus" use (≪1M vectors), skip approximate indexes. Maintain only a `Map<id, Float32Array>`. For search: stream every vector, maintain a sorted top-K window, swap in only if score > current min. Sort ascending so `results[0]` is always the min.
- **Where.** `src/state/vector-index.ts`.
- **Re-impl.** ~30 LOC. Inflection point is around 100k vectors at 1024 dim — if we cross that, switch to HNSW. The pattern's value is *"do the dumb thing until you must do otherwise."*

### A8. Vector-dimension corruption guard at restore time  [H]
- **Idea.** Embedding providers change. The on-disk index might have mixed dimensions or a dimension that no longer matches the live provider. Cross-dim cosine returns 0, silently degrading recall with no error. Solution: walk every persisted vector, return list of `{id, dim}` mismatches plus the set of distinct dims seen. Refuse to start the service if any mismatch is found, unless an explicit env flag opts in to "drop stale index, rebuild from live observations."
- **Where.** `vector-index.ts::validateDimensions`, used by `src/index.ts` startup.
- **Re-impl.** Tiny method, big payoff. Anywhere we persist embeddings and the dimension can drift, copy this guard idea (not the code).

### A9. Cross-encoder reranker invocation format & truncation  [L]
- **Idea.** Pair text is `"${query} [SEP] ${title} ${narrative}".slice(0, 512)`. Truncating at the character level (not token) is sloppy but predictable; the SEP marker lets the model do the cross-attention. Worth noting: they don't rerank when results.length ≤ 1 — explicit short-circuit avoids cost on degenerate cases.
- **Where.** `reranker.ts`.

### A10. Per-key promise-chain mutex (keyed serialization)  [H]
- **Idea.** Trivially small primitive (~15 LOC) that serializes async operations per key without locks or queues. Three subtle parts:
  1. **Chain head:** `next = (locks.get(key) ?? Promise.resolve()).then(fn, fn)`. The duplicate `fn` (used as both fulfill *and* reject handler) means `fn` runs whether the previous link resolved or rejected — without it, a single failed op would block all subsequent ops for that key forever.
  2. **Cleanup tail:** a separate `cleanup = next.then(noop, noop)` that swallows rejections — so the value stored in the map can be safely awaited by the next caller without becoming `unhandledRejection`. The map stores `cleanup`, not `next`.
  3. **Compare-and-delete:** `if (locks.get(key) === cleanup) locks.delete(key)`. Only delete if a newer chain hasn't already replaced ours — without this guard, a fast follow-up call would have its chain dropped from the map and run unserialized.
- **Where.** `src/state/keyed-mutex.ts`.
- **Re-impl.** Highest-leverage utility to copy as an *idea*. We probably want our version to optionally include error logging in the catch-side; theirs is silent.

---

## B. Memory operations

### B1. Retention scoring: salience × exp(-λΔt) + Σ σ/days_since_access  [H]
- **Idea.** Three-component retention score, combined as `min(1, salience × temporalDecay + reinforcementBoost)`:
  - **Salience** — start from 0.5 base, look up type-weighted override: `architecture=0.9, preference=0.85, pattern=0.8, bug=0.7, workflow=0.6, fact=0.5` (unknown types fall back to 0.5). Add access-count bonus: `min(0.2, accessCount × 0.02)`. For semantic memories with a `confidence` field, take `max(baseSalience, confidence)` before adding the bonus. Final salience clamped to ≤ 1.
  - **Temporal decay** — Ebbinghaus-style: `exp(-λ × days_since_created)`.
  - **Reinforcement boost** — `σ × Σ (1 / days_since_each_access)` summed over a bounded recency window (see B15, capped at 20 entries). Recent accesses pump the score back up; an access 1 day ago contributes σ × 1.0, a 10-day-old access contributes σ × 0.1. Skip access timestamps where `daysSinceAccess <= 0` (don't divide by zero on near-now accesses).
- **Defaults.** `λ = 0.01`, `σ = 0.3`. Tier thresholds `hot = 0.7, warm = 0.4, cold = 0.15`. Anything below `cold` is "evictable/frozen".
- **Config validation.** Reject λ ≤ 0 or non-finite; reject σ < 0 or non-finite; require `hot ≥ warm ≥ cold ≥ 0`. Surface the violation as a clear error string at config parse time, not at score time.
- **Where.** `src/functions/retention.ts`.
- **Re-impl.** The split into base/decay/reinforcement is the part worth keeping — many memory systems collapse to a single counter, which can't be tuned.

### B1b. Tier-specific decay rates `strength *= 0.9^(daysSince/decayDays_per_tier)`  [M]
- **Idea.** Different memory tiers (episodic / semantic / procedural) consolidate and decay at different speeds; encode that as **a per-tier `decayDays` parameter** plugged into the same exponential. Semantic memory weighs confidence too. Procedurals are only minted from patterns observed ≥2 times.
- **Where.** `src/functions/consolidation-pipeline.ts`.
- **Re-impl.** A 5-line `decayMultiplier(tier, daysSince)` lookup with a small config map. Far more expressive than one global λ.

### B2. Auto-forget pipeline (TTL + Jaccard contradiction + low-value sweep)  [H]
- **Idea.** Periodic background sweep does three independent forgets, in order:
  1. **TTL expiry** — anything with `forgetAfter` past now → hard delete. Decrement any imageRef before deleting. Track deleted ids in a `Set` so step 2 doesn't redo them.
  2. **Contradiction detection** — Jaccard over content tokens; pairs **strictly above** 0.9 (`sim > 0.9`, not ≥) mark the *older* of the pair as `isLatest=false` (soft delete only — don't hard-delete contradictions). Important details:
     - Restrict to `isLatest !== false` memories, take top 1000 by recency to bound work.
     - Token set = `lowercase(content).split(/\s+/)` filtered to tokens with length > 2.
     - Bucket memories by **concept tag** (lowercased) via a `conceptIndex: Map<concept, memId[]>`. Compare only pairs within a shared concept bucket.
     - Use a `compared: Set<string>` of canonical-ordered keys `"${minId}|${maxId}"` to skip pairs that share *two* concepts (otherwise they'd be compared twice). This is the crucial guard — without it, the inner loop revisits the same pair through every shared concept and your O(n × b) becomes O(n × b × c).
     - Pair similarity: `intersection / (|A| + |B| - intersection)`. Skip when either set is empty.
  3. **Low-value observations** — for each session, walk its observations; if `age > 180 days AND (importance ?? 5) <= 2`, hard-delete. Iterate sessions in chunks of 10 in parallel. On delete, decrement `imageData`/`imageRef` refs (avoid double-decrement when they're equal).
- **Audit.** Every deletion writes an audit record naming the reason (`"auto-forget TTL"`, `"auto-forget contradiction"`, `"auto-forget low-value observation"`). We should do the same for anything destructive.
- **Where.** `src/functions/auto-forget.ts`.
- **Re-impl.** The concept-bucketing + `compared` Set is the gem — turns O(n²) into ~O(n × avg_bucket_size). Don't skip either half.

### B3. Crystallize: LLM-summarize a chain of actions into a digest + mint lessons  [M]
- **Idea.** After a chain of completed actions, ask the LLM (single call, structured JSON output) to produce `{narrative, keyOutcomes, filesAffected, lessons}`. Then store the digest as a "crystal" and *also* fan-out the `lessons` array as separate persisted "lesson" records (each with confidence 0.6, sourced back to the crystal). The crystal's source actions get a back-pointer (`crystallizedInto`).
- **Where.** `src/functions/crystallize.ts`.
- **Re-impl.** The pattern is "one LLM call yields N indexed sub-artifacts." Generic and useful for any post-task reflection step.

### B4. Frontier scoring: priority × 10 + age + dependency-graph status  [M]
- **Idea.** "What should the agent do next?" — `frontier` walks all open actions, filters out anything blocked, scores the rest, returns top-K. The `next` endpoint is just `frontier(limit=1)`.
- **Blocker filter.** Drop any action where ANY of:
  - An outbound `requires` edge points to an action whose status is not `done`.
  - An outbound `gated_by` edge points to a checkpoint whose status is not `passed`.
  - A `conflicts_with` edge (in either direction) involves another action currently `active`.
  - An active, unexpired `lease` exists for this action under a different `agentId` (skip unless `includeLeasedByOthers`).
- **Scoring formula** (computed only on unblocked actions):
  ```
  score = priority * 10
        + min(ageHours * 0.5, 20)               // age decay, capped to prevent starvation
        + unlockCount * 5                       // count of outbound "unlocks" edges from this action
        + (has any outbound "spawned_by" edge ? 3 : 0)
        + (action.status === "active"          ? 15 : 0)
  ```
  Round to 2 decimals before returning.
- **Edge type vocabulary.** `requires | gated_by | conflicts_with | unlocks | spawned_by`. Encode as a string field on a flat `ActionEdge` row; in-memory `Map<actionId, Action>` for lookups.
- **Where.** `src/functions/frontier.ts`.
- **Re-impl.** Don't reach for a real graph DB — `kv.list` + in-memory map is fine up to thousands of nodes. The age cap (20) is the non-obvious knob; without it old low-priority actions slowly outscore fresh high-priority work.

### B5. Per-tool-call dedup with sliding TTL + sampled cleanup  [H]
- **Idea.** `DedupMap` hashes `(sessionId, toolName, JSON.stringify(input).slice(0,500))` with SHA-256, stores `{hash, expiresAt}`, returns `isDuplicate(hash)`. TTL = 5 min, cleanup runs every 60s (interval is `unref`'d so it doesn't keep the process alive). The 500-char input cap prevents pathological inputs from blowing up dedup memory.
- **Where.** `src/functions/dedup.ts`.
- **Re-impl.** Trivial, copy the **idea** anywhere we need to deduplicate streaming events. The 500-char input cap is the non-obvious detail.

### B6. Two LLM-free compression paths + one LLM path  [M]
- **Idea.** Tools that record observations have THREE compression strategies:
  - `compress-synthetic.ts` — programmatic title/narrative from tool args (zero LLM tokens). Uses regex over tool names + a small fixed list of common key names (`file_path`, `filepath`, `path`, etc.) to extract the salient field. Narratives are truncated at 400 chars — predictable, lossy, free.
  - `flow-compress.ts` — batched LLM compression on a window.
  - `compress.ts` — single-call LLM summarize. Wrapped with a hard cap (e.g. 10 invocations per sweep) and a 30s per-call timeout so a misbehaving provider can't stall the sweep.
  Default is synthetic; LLM is opt-in via env flag. The principle: **don't burn user tokens by default** for housekeeping.
- **Where.** `src/functions/compress*.ts`, `consolidate.ts`.
- **Re-impl.** Architectural principle to internalize: any background task that touches an LLM provider should default OFF and have a clear cost warning at boot. Always pair LLM calls with (a) per-sweep invocation cap and (b) per-call timeout. See also pre-tool hook below.

### B7. Decay sweep cadences as opt-out env switches  [L]
- **Idea.** Two long-running setInterval timers: `lesson-decay-sweep` every 24h, `insight-decay-sweep` every 24h, plus `auto-forget` every hour. Each is gated by an env var that defaults to enabled but can be disabled with `=false`. All timers `unref()`'d.
- **Where.** `src/index.ts` bottom.

### B8. Branch-aware memory isolation  [L]
- **Idea.** Memories carry a `branch` field. Search/list scopes can filter to current git branch. Prevents context bleed between feature branches. Branch detection uses `execFile("git", ["rev-parse", ...])` with a 5s timeout; differentiates worktrees by comparing `gitDir` vs `commonDir`, parses worktree state from porcelain output.
- **Where.** `src/functions/branch-aware.ts`.
- **Re-impl.** Thin git wrapper via `execFile` (with timeout) is much lighter than nodegit/isomorphic-git. The `gitDir ≠ commonDir` test for worktree detection is the non-obvious bit.

### B9. Working-memory two-tier budgeting (core 30% / archival 70%)  [H]
- **Idea.** A token-budgeted working set splits its budget into **core** (30%, pinned + high-score) and **archival** (70%, longer tail). Each entry scored `0.5 × importance + 0.3 × recency + 0.2 × access_count`. Auto-page evicts the *lowest-scored unpinned* entries when over budget. Pinning is an explicit hard flag, not a high score.
- **Where.** `src/functions/working-memory.ts`.
- **Re-impl.** Cleaner than a single ranked window. Core slot is the "always present" anchor; archival is the "best-effort context." Pinning prevents critical state from being scored out.

### B10. Lesson confidence: asymptotic reinforcement + dual-condition soft-delete  [M]
- **Idea.** When a lesson is re-observed, `confidence += 0.1 × (1 - confidence)` — bounded approach to 1.0, can never overshoot. Decay sweep: `confidence -= decayRate × weeks_since_baseline`. Soft-delete *only when both* `confidence ≤ 0.1` AND `reinforcements == 0` — preserves never-reinforced lessons that decayed but might still be valid.
- **Where.** `src/functions/lessons.ts`.
- **Re-impl.** Both halves matter. Asymptotic reinforcement is the standard "Brownian confidence" trick; dual-condition delete is the smart part.

### B11. Skill extraction with fingerprinted dedup + frequency reinforcement  [M]
- **Idea.** Mine reusable procedures from completed sessions via LLM. Soft-fail if the session is exploratory (LLM returns a sentinel like `<no-skill/>`) or has fewer than 2 steps after parsing.
- **Fingerprint algorithm** (the heart of dedup):
  ```
  payload = JSON.stringify({
    title:   title.toLowerCase(),
    trigger: trigger.toLowerCase(),
    steps:   steps.map(s => s.toLowerCase().trim())  // order-preserving
  })
  id = "skill_" + sha256(payload).slice(0, 16)        // 64-bit hex truncation
  ```
  Same title + same trigger + same step sequence (modulo case/whitespace) → same id → collision = dedup. Different step *order* → different id (intentional; order matters for procedures).
- **Reinforcement on hit.** When a fingerprint already exists in the procedural store:
  - If `sourceSessionIds` already contains the current `sessionId`, do nothing except bump `updatedAt` (don't reinforce on re-run of the same session).
  - Otherwise: `strength = min(1.0, strength + 0.15)`, `frequency++`, append the new sessionId. No LLM call needed.
- **Initial mint.** New skill stored with `strength = 0.6`, `frequency = 1`, source sessions/observations tracked (cap observation refs at 10).
- **Where.** `src/functions/skill-extract.ts`.
- **Re-impl.** Anywhere we mine "patterns" from history: LLM extract → structured fingerprint → frequency-based reinforce-without-re-extract is the right shape. The "same session can't reinforce twice" guard prevents repeated cron runs from inflating strength.

### B12. Reflect: BFS-on-concept-graph clustering with Jaccard fallback  [M]
- **Idea.** To cluster related memories: primary strategy is BFS over the concept co-occurrence graph (depth ≤ 2, capped at 20 clusters). If the concept graph is sparse (newly bootstrapped system), fall back to Jaccard similarity (≥ 0.3 threshold) on term-doc sets. Top 50 insights returned.
- **Where.** `src/functions/reflect.ts`.
- **Re-impl.** Pattern: primary structural algorithm + dumb-but-reliable fallback. Same shape used elsewhere (CJK, reranker). Codify it as a design principle.

### B13. Sliding-window enrichment with lookback/lookahead  [M]
- **Idea.** For each observation, expose a temporal context (default `lookback=3`, `lookahead=2` adjacent observations) and ask the LLM to resolve pronouns to explicit entities, lift preferences, add bridges to the prior/next observations. Soft-fail returns `null` if no context — never errors. Makes individual observations self-contained for later retrieval.
- **Where.** `src/functions/sliding-window.ts`.
- **Re-impl.** Useful concept: enrich observations *after* the fact using a small temporal window, so when they're retrieved out of order they still make sense.

### B14. Query expansion: reformulations + temporal concretizations  [M]
- **Idea.** LLM generates 3–5 paraphrased query reformulations *and* converts relative-time phrases ("last week") into ISO date ranges. Entity extraction via simple heuristics (quoted strings, Capitalized Words). Soft-fails to empty array on parse error rather than throwing.
- **Where.** `src/functions/query-expansion.ts`.
- **Re-impl.** Pattern: don't trust the user query as the only retrieval key. Expand cheaply, parallel-search, merge results, take max-score per id (see A1).

### B15. Access tracker: bounded recency window + Set-based batch dedup  [M]
- **Idea.** Per-memory `AccessLog` keeps a `recent` array of timestamps **capped at 20 entries**. Batch updates dedup by `Set<memoryId>` before issuing parallel writes. The cap matters: reinforcement formulas integrate over `recent`, so unbounded growth would corrupt scores AND waste storage. Keyed mutex (A10) serializes concurrent updates per memory id.
- **Where.** `src/functions/access-tracker.ts`.
- **Re-impl.** Combine with A10 + B1. Capping recency to a fixed window is the easy detail to miss.

### B16. Sentinels: pluggable event-driven gatekeepers  [L]
- **Idea.** A "sentinel" is a typed trigger config (`webhook | timer | threshold | pattern | approval | custom`). Pattern triggers regex on observation titles; threshold triggers compare metrics with operators (`gt | lt | eq`). Fires a downstream function when its condition matches. Decoupled, schema-typed, composable.
- **Where.** `src/functions/sentinels.ts`.
- **Re-impl.** Useful as a *concept* for event-driven memory hooks but probably over-engineered for our scope.

### B17. Temporal graph with closed-enum entity and relation types  [L]
- **Idea.** LLM extracts entities + relationships into fixed taxonomies so downstream code can pattern-match safely. Open-ended NER yields drift; closed enums yield reliable JSON/XML.
- **Entity types (13):** `file, function, concept, error, decision, pattern, library, person, project, preference, location, organization, event`.
- **Relation types (16):** `uses, imports, modifies, causes, fixes, depends_on, related_to, works_at, prefers, blocked_by, caused_by, optimizes_for, rejected, avoids, located_in, succeeded_by`.
- **Edge metadata.** Each edge carries: `weight ∈ [0.1, 1.0]` (1.0 = explicit, 0.5 = inferred, 0.1 = speculative), `valid_from` / `valid_to` ISO dates (or `"unknown"` / `"current"` sentinels), `reasoning`, `sentiment ∈ {positive, negative, neutral}`, optional list of `alternatives` that were considered.
- **Versioning.** Never overwrite a relation — when a new edge would conflict with an existing `(source, target, type)` triple, mark the old one `isLatest=false`, set its `tvalidEnd`, link `supersededBy = newEdgeId`, and bump the new edge's `version`.
- **Where.** `src/functions/temporal-graph.ts`.
- **Re-impl.** When asking an LLM for structured extraction, **constrain the type space** to a closed enum. The taxonomies here are tuned for coding-agent memory; pick your own — but pick a *finite* list.

---

## C. Infra / system patterns

### C1. Constant-time HMAC token comparison  [H]
- **Idea.** Auth compares Bearer tokens through `timingSafeEqual` on the **HMAC-SHA256** of each side (with a process-random key). Even if the tokens are different lengths, you compare fixed-length digests — `timingSafeEqual` throws on length mismatch otherwise. The HMAC key is `randomBytes(32)` per process; never persisted.
- **Where.** `src/auth.ts`.
- **Re-impl.** ~10 LOC. Use this anywhere a string is compared against a secret. Plain `===` leaks length-of-prefix-match timing.

### C2. CSP nonce builder for the viewer / dashboard  [M]
- **Idea.** Strict per-request CSP. Each response mints a fresh nonce, computes the `Content-Security-Policy` header that embeds it, and string-replaces a fixed placeholder in the static HTML template with the same nonce.
- **Recipe:**
  1. Define a single placeholder constant (e.g. `__APP_VIEWER_NONCE__`). Bake the same string into every `<script nonce="__APP_VIEWER_NONCE__">` attribute in the HTML template.
  2. Per request: `nonce = randomBytes(16).toString("base64url")`.
  3. Send: `html.replaceAll(placeholder, nonce)` as the body, with header `Content-Security-Policy: default-src 'none'; script-src 'nonce-${nonce}'; script-src-attr 'none'; connect-src 'self' http://127.0.0.1:* http://localhost:*; ...`.
  4. Never include `'unsafe-inline'` or `'unsafe-eval'` in `script-src`.
- **Where.** `src/auth.ts::buildViewerCsp` + `VIEWER_NONCE_PLACEHOLDER` constant.
- **Re-impl.** Anywhere we serve local HTML, mint a per-response nonce and lock CSP. Tightening `default-src` to `'none'` and explicitly allow-listing each directive is more secure than `'self'` with carve-outs.

### C3. Fail-open hooks: timeout, swallow errors, never block parent process  [H]
- **Idea.** Hooks that call back into the memory service do:
  ```
  fetch(url, { signal: AbortSignal.timeout(2000) })
    .then(... write response to stdout ...)
    .catch(() => { /* don't block tool execution */ });
  ```
  The early `if (!INJECT_CONTEXT) return;` is even more aggressive — when the feature is off, the hook doesn't even open stdin.
- **Where.** `src/hooks/*.ts`.
- **Re-impl.** Universal rule: hooks called by an outer harness must never throw past their own boundary, must have a hard timeout, must early-return when disabled before touching any I/O.

### C4. Recursion guard via dual signal (env var + payload entrypoint)  [H]
- **Idea.** When an LLM-backed memory service is invoked by its own child process (e.g. an SDK summarize call), it must not summarize again recursively. The hook checks BOTH `process.env.AGENTMEMORY_SDK_CHILD === "1"` and `payload.entrypoint === "sdk-ts"`. Two independent signals — either alone is spoofable; together gives high confidence.
- **Where.** `src/hooks/sdk-guard.ts` (per agent report), checked in `pre-tool-use.ts`.
- **Re-impl.** Pattern: when an inner-loop guard is critical, prefer two cheap independent signals over one elaborate one.

### C5. Deferred boot-log buffer + bounded capacity  [M]
- **Idea.** Boot lines (`[agentmemory] X enabled`, ~25 of them) are buffered into an in-memory array capped at 500 entries by default; only flushed to stderr in verbose mode. CLI surfaces a compressed summary instead. Logger functions `try/catch` around `process.stderr.write` so a broken stderr cannot crash a handler.
- **Where.** `src/logger.ts`.
- **Re-impl.** Two ideas: (a) suppress noisy boot output by default — show only on `--verbose`; (b) every log call is `try/catch` to insulate handlers from logging failures.

### C6. Noop-counter / noop-histogram instrumentation pattern  [H]
- **Idea.** Telemetry surface defines `Counter.add(n)` and `Histogram.record(v)` interfaces. If OTEL is wired, real counters; otherwise `NOOP_COUNTER`/`NOOP_HISTOGRAM` whose methods are empty. Call sites never branch — they just `metrics.foo.add(1)` unconditionally.
- **Where.** `src/telemetry/setup.ts`.
- **Re-impl.** Anywhere observability is opt-in, this pattern lets us instrument freely without `if (metrics)` checks at every call site. We've half-done this elsewhere; do it fully.

### C7. JSON-RPC 2.0 strict transport: don't reply to notifications, validate id  [M]
- **Idea.** Two correctness rules that bite naive implementations:
  - **Notifications** (requests with no `id` field) MUST NOT receive a response. Strict clients close the connection on spurious replies.
  - `id` must be string | number | null only. Malformed ids (object, array, bool) trigger an Invalid Request error response with `id: null` since echoing back a non-spec id is its own violation.
- **Where.** `src/mcp/transport.ts`.
- **Re-impl.** Anywhere we implement JSON-RPC, codify these as test cases. Most homegrown implementations get notifications wrong.

### C8. Debounced index persistence with throttled failure logging  [H]
- **Idea.** Index writes call `scheduleSave()`. A 5-second `setTimeout` coalesces N writes into one flush. The `setTimeout` callback wraps the save promise to consume rejections (would otherwise become `unhandledRejection`). Persistence failures log once per 60s — bursts under load only emit one line, with a `code === "TIMEOUT"` hint differentiating queue pressure from real errors.
- **Where.** `src/state/index-persistence.ts`.
- **Re-impl.** The pattern is "debounce + swallow promise + throttled error". All three pieces are necessary; missing the promise swallow means the process crashes under sustained write load.

### C9. Top-level `unhandledRejection` net with rate-limited log  [M]
- **Idea.** Long-lived service can't afford a single timed-out KV write to kill the worker. Top-level `process.on('unhandledRejection', ...)` log-and-continues, with a 60s rate limit on the warning so bursts don't spam stderr. Important: this is **after** every call site already has its own `.catch()` — the global handler is a safety net for "I missed one," not a primary strategy.
- **Where.** `src/index.ts` near the top of `main`.

### C10. Provider fallback chain with explicit noop terminus  [M]
- **Idea.** Provider lookup probes env vars in priority order: OpenAI-compatible → Anthropic → Gemini → OpenRouter → noop. Whichever has its key set first wins. Noop terminus means missing keys *degrade features* (compression returns input unchanged) rather than crash. Critical guard: agent-sdk provider requires explicit `AGENTMEMORY_ALLOW_AGENT_SDK=true` to prevent recursive summarize loops.
- **Where.** `src/config.ts`, `src/providers/index.ts`.

### C11. Default-OFF for token-burning features  [H]
- **Idea.** Two killer features were silently expensive: auto-compress (LLM-summarized every observation) and inject-context (PreToolUse hook injected ~4000 chars into every tool call). Both *defaulted ON* in early versions; one user complaint cycle later, both became opt-in with prominent boot warnings showing the env var name and the issue link. The boot warnings literally explain what gets billed.
- **Where.** `src/index.ts` boot output, `src/hooks/pre-tool-use.ts`.
- **Re-impl.** Principle: when a feature can drain user tokens or money proportionally to activity, it must default off. Add a boot-time warning that names the env switch.

### C12. Whitelist payload fields at REST→trigger boundary  [H]
- **Idea.** API endpoints never pass `req.body` directly to `sdk.trigger`. Every endpoint explicitly extracts and validates only the fields it expects. Documented in `AGENTS.md` as a hard rule. Catches injection attacks where a client posts extra fields hoping they're forwarded.
- **Where.** `src/triggers/api.ts` and the design rule in `AGENTS.md`.
- **Re-impl.** Hard rule for any internal RPC: callers serialize a whitelisted DTO, never `...req.body`.

---

## D. Linguistic & tokenization

### D1. Porter Stemmer (English) compact implementation  [L]
- **Idea.** ~100 LOC two-step suffix-rule stemmer with measure-of-roots (vowel/consonant pattern), CVC-doubling rules, two suffix maps (step2map and step3map). Plain Porter algorithm, no Snowball, no language detection. Public-domain algorithm.
- **Where.** `src/state/stemmer.ts`.
- **Re-impl.** Don't write our own; pull a small, audited library. Mention it here because some of our scripts are doing dumb `.toLowerCase()` term matching and could benefit.

### D2. CJK segmentation with lazy optional-dep fallbacks  [M]
- **Idea.** Detect script via Unicode property regex (`\p{Script=Han}`, etc.), then lazily `await import()` script-specific segmenters (jieba for Chinese, tiny-segmenter for Japanese, regex for Korean). If the optional dep isn't installed, fall back to returning the whole token and emit a *deduped one-time* warning via a `Set<string>` of already-warned keys. Non-CJK runs pass through `stem()`.
- **Where.** `src/state/cjk-segmenter.ts`.
- **Re-impl.** Pattern: optional heavyweight deps + lazy dynamic import + one-time warning sets. Generalizes to "this feature works better with X, gracefully degrades without."

---

## E. Process / repo hygiene

### E1. "Consistency rules" checklist in AGENTS.md  [M]
- **Idea.** A maintained list in `AGENTS.md` that says *"when you add an MCP tool, you MUST update these 7 places."* Forces a procedural memory across edits where the cross-file invariants would otherwise drift. Includes test count assertions and README counts.
- **Where.** `AGENTS.md`.
- **Re-impl.** Anywhere we have one-to-many fan-out (registering a thing in N files), maintain an explicit checklist in CLAUDE.md or AGENTS.md.

### E2. Boot logs that explain WHY a default is what it is  [L]
- **Idea.** When booting with a feature OFF, the boot message includes the issue number and the rationale: `"Auto-compress: OFF (default, #138) — observations indexed via zero-LLM synthetic compression. Set AGENTMEMORY_AUTO_COMPRESS=true to opt-in"`. It teaches the operator on every boot.
- **Where.** `src/index.ts`.

---

## F. Things explicitly NOT worth harvesting

- **`DESIGN.md` Lamborghini theme** — a maximalist black/gold design system tutorial. Pure prose, no extractable logic. Skip.
- **53-tool MCP surface** — useful as a *taxonomy reference* for what memory operations can exist, but the breadth is itself a smell (most tools are barely-used). Don't replicate the surface area; pick the 5-10 ops you actually need.
- **Top-level `unhandledRejection` swallow** — only because the *underlying issue* is iii-engine SDK timeouts under load. We don't have that infrastructure, so the safety net is solving a problem we don't have. **Pattern is generic and good (C9)**, but its presence here isn't itself a vote for it.
- **Brute-force vector search (A7)** — fine as a starting point; do NOT carry the "we'll add HNSW later" comment into production without a real threshold.
- **Hand-curated synonym list** — domain-locked to coding-agent memory. We'd need our own.

---

## Priority summary

**High (worth doing soon if relevant):**
A1 RRF + adaptive weights · A3 lazy fail-open reranker · A8 dimension-corruption guard · A10 keyed mutex · B1 retention scoring formula · B2 auto-forget with concept-bucketed contradiction detection · B5 dedup with input-size cap · B9 working-memory two-tier · C1 constant-time HMAC compare · C3 fail-open hooks · C4 dual-signal recursion guard · C6 noop-instrumentation · C8 debounced persistence with throttled failure log · C11 default-OFF for token-burning features · C12 whitelist DTO at boundary

**Medium:**
A2 graph expansion · A4 session diversification · A5 BM25 prefix expansion · A6 synonym half-weight · B1b tier-specific decay · B3 crystallize as fan-out · B4 frontier scoring · B6 LLM-free compression default + caps · B10 lesson asymptotic confidence · B11 skill fingerprint reinforcement · B12 BFS-with-Jaccard-fallback clustering · B13 sliding-window enrichment · B14 query expansion + temporal concretization · B15 bounded access log · C2 CSP nonce · C5 boot buffer · C7 JSON-RPC strict · C9 unhandled rejection net · C10 provider fallback · D2 lazy CJK · E1 consistency checklist

**Low:**
A9 reranker pair format · A7 brute-force vector · B7 decay cadences · B8 branch-aware · B16 sentinels · B17 typed-entity extraction · D1 Porter stemmer · E2 issue-numbered boot logs
