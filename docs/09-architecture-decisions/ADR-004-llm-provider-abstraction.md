# ADR-004: Provider-agnostic LLM client; Ollama default for local dev, Anthropic for cloud dev, Azure OpenAI as production target

- **Date:** 2026-05-08 (revised 2026-05-08 — see Revisions)
- **Status:** Accepted

## Context

Phase 3 introduces LLM calls. The challenge brief names Azure OpenAI as the
production target; the project's working environment ships an Anthropic API
key. Two non-trivial decisions follow from that gap:

1. **What contract do agents call?** Every Phase 3 worker agent
   (triage / retrieve / reason / clarify / verify) plus the supervisor calls
   an LLM at least once. If each agent imports a vendor SDK directly, swapping
   providers is a refactor; if they share a thin abstraction, the swap is a
   factory change.
2. **What ships today versus tomorrow?** A complete Azure backend would
   require live credentials we cannot exercise in this environment. A
   partial-but-untested Azure path is worse than no path at all because it
   masks a runtime failure as a code-complete deliverable.

Constraints:

- Acceptance gate (`docs/ROADMAP.md`, issue #17): integration test against
  the live Anthropic API gated by `ANTHROPIC_API_KEY`; unit tests use a
  mock.
- Acceptance gate (issue #18): "Full implementation if `AZURE_OPENAI_*`
  keys present, otherwise a documented stub raising `ConfigurationError`
  with a clear message."
- Architecture principle (`CLAUDE.md`): "Pydantic at every boundary." No
  raw `dict` between agents — and by extension, no provider-native message
  types either.
- Architecture principle (`CLAUDE.md`): "Deterministic core; LLM at the
  edges." Tests must be reproducible without network access.

## Decision

The LLM layer is a single `Protocol` with three backends (two
implemented — Ollama and Anthropic — plus the Azure documented stub) and
a deterministic mock; provider selection is an env-var flip.

### 1. `LLMClient` is a `typing.Protocol`, not a class hierarchy

`src/wscad_triage/llm/client.py` defines:

```python
@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse: ...
```

A `Protocol` over an abstract base class because (a) backends share no
implementation worth inheriting; (b) `MockLLMClient` does not need an
import of the real backends to satisfy the contract; (c) duck-typed
satisfaction lets test fakes appear inline without inheritance ceremony.

The Protocol's surface is intentionally narrow:

- `messages: list[Message]` — provider-agnostic conversation.
- `tools: list[ToolSpec] | None` — declarations the model may call.
- `response_format: type[BaseModel] | None` — signals "structured output."
  Anthropic does not have a native equivalent of OpenAI's
  `response_format`; the convention is that callers wanting structured
  output declare a `ToolSpec` whose `input_schema` is the target Pydantic
  model's JSON schema and inspect the resulting `tool_calls`. The
  parameter is preserved in the signature for ROADMAP fidelity (#17) and
  to leave a hook for backends that *do* support native structured output
  in the future.

Per-call vendor-specific knobs (`max_tokens`, temperature, stop sequences)
live on the backend constructor, not the Protocol. Putting them all on
the Protocol would re-bind the abstraction to a particular provider's
flavour; putting one on (e.g. `max_tokens`) and excluding others would
be arbitrary. Backends own their own request shape.

### 2. Provider-agnostic message types via `wscad_triage.llm.types`

`Message`, `ToolSpec`, `ToolCall`, `LLMResponse`, `Usage` are Pydantic
models with the ADR-002 conventions (`extra="forbid"`, `Literal` for
roles and stop reasons, bounded `int` token counts). Backends translate to
and from provider-native shapes inside their module; nothing else in the
codebase imports `anthropic` or `openai`.

This is the boundary type for the LLM layer. The cost is a thin translation
layer in each backend. The benefit is that adding a third provider, or
swapping a model family, never touches an agent.

### 3. Three backends; provider chosen via `WSCAD_TRIAGE_PROVIDER`

- **Ollama (default)** — self-hosted, runs against a local server.
  `OllamaBackend` (issue #55) is the default because the pipeline must be
  runnable without a paid API account: clone the repo,
  `ollama pull gpt-oss:20b`, and the workers' tool-use round-trips work.
  Tested model: `gpt-oss:20b`. Tool-use is supported but model-dependent
  — smaller open-weight models (qwen2.5:7b, llama3.1:8b, mistral:7b) can
  emit malformed JSON arguments. The risks doc maintains the
  recommended-model list.
- **Anthropic** — cloud, paid. `AnthropicBackend` is the cheapest cloud
  path because of prompt caching: any `Message` with `cache=True` is sent
  with `cache_control={"type": "ephemeral"}` on its content block. Agents
  put the (large, stable) KB chunks behind a cache marker and the
  (small, per-request) ticket text after it.
- **Azure OpenAI** — production target, stub today. `AzureOpenAIBackend`
  raises `ConfigurationError` on construction redirecting the operator
  to the two working providers (`ollama` default, `anthropic` if a paid
  key is available). The class still implements the Protocol so the
  factory's type-narrowing path stays clean; failure is at startup, not
  mid-pipeline. Full implementation is tracked as future work — the
  acceptance criterion in the ROADMAP is met by the documented-stub
  branch of issue #18.

### 4. `MockLLMClient` for tests; no silent fallthrough

`MockLLMClient(script={"trigger": LLMResponse(...) | [LLMResponse(...), ...]})`
matches the first scripted *trigger substring* against the concatenated
message contents. Unmatched calls raise `RuntimeError`; per-trigger lists
support multi-turn flows by consuming the next response on each match.
Each call captures `(messages, tools, response_format)` so tests can
assert tool advertisement and structured-output requests, not only text.

The strict failure mode catches one common bug class — an agent silently
routing to the wrong tool — but is shallow defence: substring matching
means any two prompts that share a trigger word collide on the
first-registered entry. Tests must pick triggers that uniquely identify
the calling agent (the convention is a role marker like
`"You are the triage agent"`). If PR2's integration scenarios start
producing collisions in practice, the implementation can move to
prefix-matching on the system message; this ADR does not prescribe.

### 5. Provider selection via `pydantic-settings`

`Settings` (in `src/wscad_triage/settings.py`) reads
`WSCAD_TRIAGE_PROVIDER`, `WSCAD_TRIAGE_OLLAMA_*`, `ANTHROPIC_API_KEY`,
`WSCAD_TRIAGE_ANTHROPIC_MODEL`, and `AZURE_OPENAI_*` from environment or
`.env`. `make_client(settings) -> LLMClient` dispatches on
`settings.llm_provider`. The Ollama branch is permissive — no
fail-fast credential check; the runtime contract is the local server's
reachability, validated lazily on the first `generate()` call. The
Anthropic branch fail-fasts when `ANTHROPIC_API_KEY` is missing.
Phase 3 keeps the config surface tight — no `config.yaml` yet; that
arrives with Phase 4 issue #27, which layers the rubric weights and
threshold knobs onto the same `Settings` object.

### 6. Backend equivalence is by Protocol, not feature parity

The three backends satisfy the same `LLMClient` Protocol but differ in
provider-specific niceties. Pinned explicitly so a reviewer reading just
one backend module knows what is and isn't equivalent across providers:

- `Message.cache=True` is **honoured by Anthropic** (sets
  `cache_control={"type": "ephemeral"}`); **silently ignored by Ollama
  and the Azure stub**. Agents must not depend on the flag being
  honoured for correctness — only for cost/latency on the Anthropic path.
- `ToolCall.id` is **provider-supplied for Anthropic** (the
  `block.id` from `tool_use` content blocks); **synthesised (UUID4) for
  Ollama** (the SDK doesn't emit one). Tests must not assert on stable
  ids; assert on uniqueness instead.
- `ToolCall.arguments` arrives as a dict in **all** backends. Ollama
  returns it as a dict natively; Anthropic returns a dict via
  `block.input`. (OpenAI-compatible APIs return a JSON-encoded string;
  if a future backend uses that wire shape, the adapter parses to a
  dict at the boundary.)
- Streaming is **unsupported on the Protocol today** for all three. The
  Future work section tracks the migration.

## Consequences

- **Positive.** Agents are provider-blind. Adding GPT-5, Mistral, or a
  local llama backend is one new file in `src/wscad_triage/llm/` plus a
  factory branch — no agent changes.
- **Positive.** Test runs are deterministic and offline. The
  `live_api`-marked tests (one per implemented backend — Anthropic, plus
  text + tool-use scenarios for Ollama) exercise the real SDKs end-to-end
  on demand; every other test uses `MockLLMClient`, so CI does not need
  an API key or a running Ollama server.
- **Positive.** Prompt caching is on the boundary type, not buried in the
  Anthropic backend. When Azure ships, the Azure backend can ignore the
  flag (no caching on that provider) without changing agent code.
- **Positive — strictness as a feature.** `MockLLMClient`'s
  `RuntimeError` on unmatched triggers is the only line of defence against
  a regression where an agent silently chooses the wrong tool — the
  integration test scenarios depend on it.
- **Negative.** The Azure stub advertises itself as `LLMClient` but does
  not run. Reviewers may interpret it as completion-by-name; the README
  "Configuration" section names the constraint explicitly to avoid that
  read.
- **Negative.** Provider-agnostic types lose vendor-specific niceties.
  Anthropic's tool-result-with-image content is not expressible in the
  current `Message` type; if a future agent needs it, the type widens.
- **Negative — Ollama tool-use fidelity is model-dependent.** The
  `gpt-oss:20b` default works in practice; smaller open-weight models
  (qwen2.5:7b, llama3.1:8b, mistral:7b) have visibly worse tool-call
  reliability and can emit malformed JSON arguments that fail the
  workers' Pydantic validation. The risks doc tracks the
  recommended-model list; Phase 5 (#34) will add a per-model
  fidelity benchmark over the labelled eval set.
- **Locked-in.** Every Phase 3 agent's signature takes `LLMClient`.
  Replacing the Protocol with an ABC or with a function-only interface is
  a refactor across `src/wscad_triage/agents/`. Anticipated; the
  alternative — agents importing SDKs directly — is worse.

## Future work

- **Full Azure OpenAI backend** when production credentials are available.
  The migration: replace the constructor body in
  `src/wscad_triage/llm/azure_openai_backend.py` with a real client;
  implement `generate` against `AzureOpenAI.chat.completions.create`;
  add `tests/integration/test_azure_live.py` alongside the existing
  Anthropic and Ollama live smokes (or factor a parametrised live test
  shared across the three). No agent or pipeline code should change.
- **Streaming.** The current Protocol returns a complete `LLMResponse`.
  When latency-sensitive UIs (Phase 6+ if any) need streaming, add a
  separate `stream` method or replace `generate` with an async iterator
  — both are local changes to the backends + the supervisor's call site.
- **Prompt-caching efficiency.** Today caching is mechanically wired but
  not tuned. Phase 5's eval harness will measure the cost gap with /
  without caching once the agent prompts stabilise.
- **Per-model tool-use fidelity benchmark for Ollama.** Phase 5 (#34)
  measures whether `gpt-oss:20b` and the smaller candidates (`qwen2.5:14b`,
  `llama3.1:8b`) all pass the five-scenario integration suite live.
  Updates the recommended-model list in the risks doc when results land.

## Revisions

- **2026-05-08** — added Ollama as a third backend (issue #55), switched
  the default provider from Anthropic to Ollama so the pipeline is
  runnable offline against a local server. ADR-004 was edited in place
  rather than superseded. Concrete enumeration of edits (Title, Decision
  intro, §3, §5, §6 new, Consequences, Future work):
  - **Title** — rewrote from "Anthropic default, Azure OpenAI as
    production target" to add Ollama as the local-dev default tier.
  - **Decision intro** — counter changed from "two real backends" to
    "three backends (two implemented + Azure stub)".
  - **§3 "Anthropic is the default; Azure OpenAI is a documented stub"**
    — rewrote in full to cover three backends; the Azure stub's
    redirect-message guidance updated to point at Ollama (default) or
    Anthropic (cloud).
  - **§5 "Provider selection via `pydantic-settings`"** — added the
    `WSCAD_TRIAGE_OLLAMA_*` env-var enumeration; added the
    permissive-vs-fail-fast distinction (Ollama lazy on first
    `generate()`, Anthropic eager on missing key, Azure eager via stub
    `__init__`).
  - **§6 (new) "Backend equivalence is by Protocol, not feature parity"**
    — pins which `Message`/`ToolCall` fields are honoured vs ignored
    across the three backends.
  - **Consequences "Negative"** — rewrote the
    "Two backends advertise themselves" bullet to "The Azure stub
    advertises itself" (only one stub now, since Ollama is
    implemented); appended a new bullet for Ollama tool-use fidelity
    being model-dependent.
  - **Future work** — appended the per-model Ollama fidelity-benchmark
    bullet (#34); rewrote the Azure migration bullet's test-update
    pointer from `tests/integration/test_anthropic_live.py` to
    `tests/integration/test_azure_live.py` (the new file the
    migration would create) and noted the sibling Anthropic + Ollama
    smokes.
  - **Unchanged: §1 (Protocol), §2 (boundary types), §4
    (MockLLMClient).** The original rationale for those sections still
    applies verbatim.
