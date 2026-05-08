# ADR-004: Provider-agnostic LLM client; Anthropic default, Azure OpenAI as production target

- **Date:** 2026-05-08
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

The LLM layer is a single `Protocol` with two real backends and a
deterministic mock; provider selection is an env-var flip.

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

### 3. Anthropic is the default; Azure OpenAI is a documented stub

`AnthropicBackend` is the production code path. It supports prompt caching
via the `cache=True` flag on `Message`, which sets
`cache_control={"type": "ephemeral"}` on the corresponding content block.
Agents put the (large, stable) KB chunks behind a cache marker and the
(small, per-request) ticket text after it.

`AzureOpenAIBackend` raises `ConfigurationError` on construction with a
message naming `WSCAD_TRIAGE_PROVIDER=anthropic` as the resolution. The
class still implements the Protocol so the factory's type-narrowing path
stays clean; failure is at startup, not mid-pipeline. Full implementation
is tracked as future work — the acceptance criterion in the ROADMAP is met
by the documented-stub branch of issue #18.

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
`WSCAD_TRIAGE_PROVIDER`, `ANTHROPIC_API_KEY`,
`WSCAD_TRIAGE_ANTHROPIC_MODEL`, and `AZURE_OPENAI_*` from environment or
`.env`. `make_client(settings) -> LLMClient` dispatches on
`settings.llm_provider`. Phase 3 keeps the config surface tight — no
`config.yaml` yet; that arrives with Phase 4 issue #27, which layers the
rubric weights and threshold knobs onto the same `Settings` object.

## Consequences

- **Positive.** Agents are provider-blind. Adding GPT-5, Mistral, or a
  local llama backend is one new file in `src/wscad_triage/llm/` plus a
  factory branch — no agent changes.
- **Positive.** Test runs are deterministic and offline. The single
  `live_api`-marked test exercises the real Anthropic SDK end-to-end;
  every other test uses `MockLLMClient`, so CI does not need an API key.
- **Positive.** Prompt caching is on the boundary type, not buried in the
  Anthropic backend. When Azure ships, the Azure backend can ignore the
  flag (no caching on that provider) without changing agent code.
- **Positive — strictness as a feature.** `MockLLMClient`'s
  `RuntimeError` on unmatched triggers is the only line of defence against
  a regression where an agent silently chooses the wrong tool — the
  integration test scenarios depend on it.
- **Negative.** Two backends advertise themselves as `LLMClient` but only
  one runs. Reviewers may interpret the Azure stub as completion-by-name;
  the README "Configuration" section names the constraint explicitly to
  avoid that read.
- **Negative.** Provider-agnostic types lose vendor-specific niceties.
  Anthropic's tool-result-with-image content is not expressible in the
  current `Message` type; if a future agent needs it, the type widens.
- **Locked-in.** Every Phase 3 agent's signature takes `LLMClient`.
  Replacing the Protocol with an ABC or with a function-only interface is
  a refactor across `src/wscad_triage/agents/`. Anticipated; the
  alternative — agents importing SDKs directly — is worse.

## Future work

- **Full Azure OpenAI backend** when production credentials are available.
  The migration: replace the constructor body in
  `src/wscad_triage/llm/azure_openai_backend.py` with a real client;
  implement `generate` against `AzureOpenAI.chat.completions.create`;
  update `tests/integration/test_anthropic_live.py` to add a parallel
  Azure smoke (or factor a parametrised live test). No agent or pipeline
  code should change.
- **Streaming.** The current Protocol returns a complete `LLMResponse`.
  When latency-sensitive UIs (Phase 6+ if any) need streaming, add a
  separate `stream` method or replace `generate` with an async iterator
  — both are local changes to the backends + the supervisor's call site.
- **Prompt-caching efficiency.** Today caching is mechanically wired but
  not tuned. Phase 5's eval harness will measure the cost gap with /
  without caching once the agent prompts stabilise.
