# AI Agents: Deterministic Code Around Probabilistic LLMs

**Core Principle:** The language model is the central probabilistic decision component in the system. The code around it is built to be as deterministic as possible: it frames that unpredictability, feeds it context, validates its proposals, and catches failures.

**Core Statement:** An AI agent is not a single model, but a software system that embeds a probabilistic model inside a controlled execution loop.

**Rule of Thumb:** The model proposes, a scorer evaluates, the code decides.

---

## 1. Control Flow & Triggers

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Event & Time Triggers | Cron jobs, webhooks, queue listeners | Run agents not just interactively in chat, but on a schedule or in response to background system events. |
| Agent Loop | `while` / `for` | Call the model in a loop until the goal is reached. |
| Termination Condition | `break` / `return` | Stop when a final answer is produced or hard limits (steps, time, budget) are reached. |
| Prompt Chaining | Pipeline / function composition | Fixed sequence where the output of step 1 is the input to step 2. |
| Plan-and-Execute | Query planner (SQL) / dry-run | Have the model produce a structured work plan before code executes it step by step. |
| Evaluator–Optimizer (Reflection) | Review loop until approval | Draft → critique → revise in a targeted loop until a quality criterion is met. |
| Routing | Classification + `switch` | Have the model classify the request; code dispatches it to the matching execution path. |
| Tool Dispatch | `switch` / dictionary lookup | Map the tool name in a model response to the actual function in code. |
| Hooks / Middleware | Interceptor, pre-/post-handlers | Code that runs automatically before/after every model or tool call (logging, validation, transformation). |
| Workflow Graph | State machine / finite automaton | Enforce allowed states and transitions explicitly in code. |
| Parallelization | `async` / fan-out, fan-in | Launch multiple model or tool calls concurrently and merge their results. |
| Speculative Execution | Speculative execution / prefetching | Preload likely data or run guardrails in parallel while the model is still generating. |
| Orchestrator–Worker | Master/worker, job queue | One model decomposes the overall task; specialized workers process the sub-tasks. |
| Streaming & Events | Generators (`yield`), event streams (SSE) | Forward tokens and intermediate steps continuously instead of blocking until completion. |
| Human-in-the-Loop | Blocking wait for approval | Require a human to confirm before critical steps proceed. |
| Interrupts & Cancellation | CancellationToken, signal handling | Cleanly abort a running loop when the user stops it or supplies new information. |
| Time-Travel / Replay | Debugger step-back, `git checkout` | Rewind to an earlier checkpoint, adjust context, and re-run from that point. |
| Clarifying Questions on Ambiguity | Interactive prompt | Pause when a request is ambiguous and ask the user instead of guessing (e.g. via an `ask_user` tool). Code detects this and pauses the loop. |
| Tool Choice (Forced Call) | Mandatory function call | Set `tool_choice` to `auto`, `required`, or a specific tool, e.g. to force a structured final response on the last step. |

## 2. Context, Memory & Tool Interfaces

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| System Prompt | Global constants / config header | Anchor ground rules, role, and boundaries at the very top of the context. |
| Prompt Templates | String formatting, template engine | Assemble prompts at runtime from fixed building blocks and variables. |
| Few-Shot Examples | Configuration / test fixtures | Demonstrate desired behavior and output format via examples in the prompt. |
| Tool Definitions | Function signature + docstring / API docs | Name, description, and parameter schema for every tool (the model sees only this — unclear docs lead to bad calls). |
| Deterministic Tools over Model Knowledge | Library function over mental math | Offload arithmetic, date logic, sorting, or code execution to tools instead of letting the model guess. |
| Tool Output Preparation | Pagination, truncation, serialization | Truncate, filter, or summarize oversized tool outputs before they blow up the context window. |
| Short-Term Memory | List with `append()` | Carry message history and intermediate steps through the current run. |
| Context Scoping (Isolation) | Variable scope (local vs. global) | Show each step only the exact slice of data it needs. |
| Context Compaction | Ring buffer, log rotation | Trim or summarize older history when the context window fills up. |
| Long-Term Memory | Database read / write | Persist knowledge across sessions. |
| Background Memory Extraction | ETL job / write-behind cache | Extract key facts asynchronously after a conversation and write them to a database. |
| RAG | Search query / database index | Retrieve relevant documents from a knowledge base before or during a run. |
| Dynamic Tool Loading (Tool-RAG) | Lazy loading / dynamic import | When hundreds of tools exist, load only the tool descriptions relevant to the current step into context. |
| Multimodal Preprocessing | Codec / deserialization (OCR, STT) | Convert PDFs, images, or audio into text, tables, or chunks before calling the model. |
| Prompt Caching | Prefix read cache | Avoid reprocessing an unchanged prompt prefix on every step (saves latency and cost). |
| Context Ordering | Memory layout / prioritization | Put critical instructions at the beginning or end; information buried in the middle of very long contexts is easier to miss. Keep stable parts at the front so prompt caching hits. |
| Chunking & Indexing | Index design | Split documents meaningfully (by section, with overlap and metadata such as source and date). Poor chunks produce poor RAG, no matter how strong the model is. |
| Hybrid Search & Query Rewriting | Full-text index + vector search, query expansion | Combine keyword search (e.g. BM25) and vector search because embeddings struggle with exact strings like part numbers. Let the model rewrite or decompose search queries. |
| Instruction Modules (Skills) | Plugins / on-demand modules | Load specialized instructions only when needed instead of stuffing everything into the system prompt. Keeps context lean. |
| External Working Memory (Notes & Task Lists) | Work log / swap file | Let the agent write plans, progress, and findings to files or a task list. Survives context compaction and very long runs. |

## 3. Input & Output Validation (Validation & Guardrails)

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Input Guardrails | Input validation / sanitization | Catch invalid, dangerous, or sensitive inputs before calling the model. |
| Constrained Decoding | Grammar / parser rules | Exclude invalid tokens during generation itself. |
| Structured Outputs | Types / schema (Pydantic, JSON Schema) | Enforce a typed data object instead of free-form text. |
| Parsing & Validation | Parser + exception handling | Parse model responses, check types and value ranges, and catch errors cleanly. |
| Output Guardrails | `if` / `else`, assertions | Check the response for content and format rules before passing it on. |
| Grounding & Citations | Pointer / foreign-key check | Require the model to back claims with source IDs; code deterministically verifies that the source and quote exist. |
| Deterministic Recalculation | Post-processing / recalculation | Recompute totals, percentages, or dates mentioned in text using deterministic code. |
| Logprobs / Confidence Check | Threshold filter | Inspect token probabilities. Caveat: not available from every provider and not a reliable hallucination detector (models are often confidently wrong). |
| LLM-as-Judge | Peer code review | A second model evaluates the response (caveat: itself probabilistic!). |
| Retry with Feedback | `try` / `except` in a loop | Feed the validation error back to the model so it can self-correct. |
| Multi-Sampling & Voting | Redundancy / majority vote | Generate multiple responses and pick the most frequent or highest-scoring one. |
| Fallback | Default value, fallback implementation | Switch to a backup model or fixed response when retries are exhausted. |
| Stop-Reason Handling | Checking exit codes / return codes | Inspect why generation stopped (finished, tool call, max tokens, refusal). Continue or reject truncated responses and handle refusals explicitly instead of treating them as normal answers. |

> **Note:** Guardrails are not just `if`/`else`. `if`/`else` describes the control flow; in practice guardrails are also implemented via policy engines, permission services, schema validators, sandboxes, or dedicated classifier models.

## 4. Security & Isolation

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Prompt Injection Defense | Delimiters / data tagging + least privilege | Mark untrusted external data in the prompt. Important: unlike SQL prepared statements, LLMs have no guaranteed separation between code and data. Real protection comes from hard restrictions on tool permissions in the harness. |
| Tool Argument Validation | Parameter validation, path-traversal protection | Check not only whether a tool is allowed, but also what arguments it is called with (e.g. file paths, amount limits, recipient domains). |
| Output Encoding | Escaping against XSS / SQL injection | Treat model output as untrusted input before rendering it as HTML/Markdown in a browser or interpolating it into queries. |
| Tenant Isolation in Memory | Multi-tenancy, separate schemas / namespaces | Strictly enforce in database/RAG code that User A's memory and documents never enter User B's context. |
| Identity Propagation (On-Behalf-Of) | OAuth scoping, row-level security | Execute data and tool calls with the exact permissions of the authenticated user. |
| Tool Permissions | Access control lists / allowlist / RBAC | Define which tools and actions are allowed in a given state. |
| Approval Before Irreversible Actions | "Are you sure?" dialog (two-phase commit) | Never send, delete, pay, or publish without explicit human approval. |
| Reversible PII Masking | Tokenization / proxy substitution | Replace sensitive data with placeholders before calling the model and restore it afterward. |
| Secrets Handling | Environment variables, vault | Never place passwords or API keys into the model context; inject them only inside tool code. |
| Sandbox | Containers, process isolation | Isolate code execution and file access from the host system. |
| Egress Filter (DLP) | Outbound firewall / allowlisting | Prevent a hijacked model from exfiltrating data via external links, webhooks, or images. |
| Budgets & Rate Limits | Counters, quotas | Enforce hard caps on cost and call counts per run and per tenant. |
| Audit Trail | Immutable security log | Record tamper-evident logs of who triggered which write or irreversible action and when. |
| Autonomy Levels | Privilege levels / `sudo` | Set per task: `read_only`, `suggest`, `act_with_approval`, or `autonomous`. Choose the level based on action risk, not trust in the model. |
| Trust Verification for Tools & MCP Servers | Dependency scanning, signed packages | Third-party tool servers can return manipulated tool descriptions or results. Use vetted sources, pin versions, and monitor tool descriptions for changes. |
| Data Privacy & Retention | Deletion policy, retention schedules (GDPR) | Define which data lands in logs, traces, checkpoints, and vector indexes, how long it is kept, and how it is completely deleted on request. |
| Regulatory Classification | Compliance controls / risk assessment | Depending on the use case, regulations such as the EU AI Act impose obligations (transparency, documentation, human oversight). Clarify the risk category early. |
| Red Teaming | Penetration testing | Deliberately attempt to trick the agent into misbehaving (injection, privilege escalation, data exfiltration) before attackers do. Turn findings into permanent eval cases. |

## 5. Reliability, Operations & Lifecycle

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Prompt Versioning | Version control (`git`), config as code | Version prompts like code so evals, metrics, and rollbacks map to an exact revision. |
| Model Version Pinning | Dependency pinning | Pin dated model snapshots so provider updates do not silently alter behavior. |
| Sampling Parameters | Runtime configuration | `temperature` and `top_p` control variance (note: even `temperature=0` is not guaranteed deterministic). |
| Model Cascading / Tiering | Tiered storage / escalation tiers (Support L1 → L2) | Try a small, fast model first; escalate to a larger model only when needed. |
| Timeouts | Timeout | Cleanly abort hanging model or tool calls. |
| Circuit Breaker & Backoff | Circuit breaker, exponential backoff | Wait with backoff on API outages or rate limits, or fail over to a secondary provider. |
| Idempotent Tools | Idempotent API endpoints | Ensure a retried tool call does not cause duplicate side effects. |
| Compensation Logic (Rollback) | Saga pattern / transaction rollback | If step 4 aborts, undo the side effects of steps 1–3. |
| Checkpointing | Persistence / snapshots | Resume long-running workflows from the last completed step after a crash. |
| Response Cache | Memoization / key-value cache | Avoid paying twice for identical requests. |
| Semantic Caching | Nearest-neighbor cache | Cache similarly phrased questions via vector similarity. Caution: requires very high thresholds + fact checking, otherwise "by June 30" and "by September 30" may collide. |
| Logging & Tracing | Logs, distributed tracing | Reconstruct what the agent did, when, how long it took, and why. |
| Cost Tracking | Metrics / monitoring | Track token usage (input, output, cache, reasoning) and cost per run. |
| Evals | Unit and regression tests | Measure statistically across realistic test cases whether a change degrades quality. |
| Shadow Mode / Canary | Dark launching / blue-green deployment | Test new prompts or models in the background against real traffic. |
| Feedback Data for Training | Telemetry → data pipeline (data flywheel) | Collect successful and corrected runs. The harness itself never mutates model weights — a separate training pipeline (e.g. fine-tuning) uses this data to train future models. |
| Durable Execution | Workflow engine / durable functions | Guarantee that a run continues across crashes, deployments, or multi-day waits. Builds on top of checkpoints. |
| Reasoning Budget | Compiler optimization level / compute budget | On reasoning models, configure how much thinking budget a step gets: high for hard planning, low for routine steps. Costs tokens and latency. |
| Batch Processing | Batch jobs / overnight runs | Process non-urgent workloads via provider batch APIs at lower cost. |
| Online Quality Monitoring & Drift | SLO monitoring / anomaly detection | Evals check before release; in production, sample and score live runs to catch quality drift or new request patterns early. |
| Global Kill Switch | Feature-flag emergency shutoff | Immediately disable an agent or specific tenants/tools centrally without redeploying. |

## 6. Multi-Agent Coordination

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Sub-Agent | Function call / microservice | Delegate a sub-task to an agent with its own isolated context. |
| Handoff (Context Transfer) | `goto` / thread ownership transfer | Hand over conversation and responsibility completely to a specialist agent. |
| Message Passing | Inter-service messaging / event bus | Let agents exchange results asynchronously via messages. |
| Shared State | Shared database, blackboard | Allow multiple agents to read and write to a shared store. |
| Locking on Shared State | Mutex / locks, optimistic concurrency | Prevent two parallel agents or workers from overwriting the same run or record concurrently (race conditions). |
| Tool / Agent Protocols (MCP, A2A) | Device driver / USB standard / OpenAPI | Standardize how tools and agents connect without bespoke glue code. |
| Ping-Pong / Deadlock Protection | Cycle detection / time-to-live (TTL) | Prevent agents from passing tasks back and forth in an infinite loop. |
| Least-Privilege Delegation to Sub-Agents | Scoped tokens / least-privilege delegation | Grant sub-agents only the permissions required for their sub-task, never more than the caller holds. |
| Typed Handoffs | API contracts / interface definitions (Pydantic) | Pass results between agents or phases as validated schemas rather than free text. Errors surface at the boundary, not three steps later. Checks structure, not factual truth. |

## 7. Scoring as an Intermediate Layer (Scoring & Thresholds)

Scoring bridges the gap between fuzzy model outputs and deterministic code: a soft question like "is this good enough?" becomes a number, and ordinary code decides via `if score > threshold`. Even at the single-token level, language models work this way: the network deterministically assigns scores (logits) to every token, and only sampling from that distribution is probabilistic.

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Token Scoring (Logits → Softmax) | Score function + `max()` / weighted sampling | The network scores every candidate token; `temperature`, `top_k`, and `top_p` shape the distribution before sampling. Constrained decoding and logprobs checks operate right here. |
| Similarity Scoring (Embeddings) | Distance metric / nearest-neighbor search | Express similarity between a query and a document, tool description, or cache entry as a number. Foundation for RAG, Tool-RAG, and semantic caching. |
| Reranking | Search engine ranking function | A second, more precise model re-orders coarse RAG hits so only the top *k* enter the context. |
| Classifier Scores | Spam filter score | Probability per category (injection, toxicity, topic routing); code blocks or routes above a threshold. |
| Rubric-Based Judge Scores | Grading rubric / scored checklist | An LLM-as-Judge assigns points against explicit criteria instead of vague impressions. Still probabilistic, so audit a sample against human ratings. |
| Best-of-N Selection | `max(candidates, key=score)` | Generate *N* candidate answers, score them, and return the best. Costs *N* times the tokens. |
| Confidence-Based Escalation | Threshold alert | On a low score, escalate to a stronger model (cascading) or to a human (human-in-the-loop). |
| Threshold Calibration | Tuning on test data (precision/recall, ROC curve) | Derive thresholds from evals on real examples rather than guessing. Every threshold trades false positives against false negatives — guessing just moves fuzziness into an arbitrary number. |
| Reward Model (Training) | Fitness function | AI labs use reward models to score outputs and shift weights toward higher scores (e.g. RLHF). Only relevant inside your own harness if you fine-tune models yourself. |
| Reward-Hacking Protection | Separation of auditor and auditee (four-eyes principle), tamper-proof tests | Models sometimes optimize for the score rather than the task (deleting or weakening tests, hardcoding outputs, hiding errors with `type: ignore`, sweet-talking a judge). Defenses: read-only scorers (`ProtectedPaths`), diff checks on test/config files, held-out tests, multiple independent signals (tests, typecheckers, linters), isolated judge, specific feedback instead of a raw number, bounded retries, human review before merge. |
| Pairwise Comparison | A/B comparison, tournament | Have a judge decide "is A or B better?" instead of assigning absolute 1–10 scores. Usually more stable. |
| Judge Bias Control | Blind review, randomization | Judges often favor longer answers, the first option shown, or outputs from their own model family. Randomize order, account for length, and use a different model family as judge. |

> **Where scoring already appears elsewhere in this taxonomy:** RAG and Tool-RAG (similarity), routing and input guardrails (classifiers), LLM-as-Judge, multi-sampling & voting, logprobs checks, model cascading, semantic caching, and evals (pass rate as release gate).

## 8. Multi-Model Operation (Model-Agnostic Architecture)

Models from different labs react differently to the same prompt, prefer different formats, and have different strengths — partly because each lab trains its models inside its own harness environment. **Principle: shared safety core, model-specific edges, evals as the referee.** A harness built on the lowest common denominator wastes model capabilities; a harness coupled to a single model creates lock-in.

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Model-Agnostic Safety Core | Server-side validation (client-independent) | Tool permissions, sandboxes, egress filters, approvals, tenant isolation, and budgets never depend on the model. A weaker model may produce worse answers, but cannot cause harm. |
| Provider Adapter | Adapter pattern / device driver | Thin layer translating the canonical internal format into each provider's API format (tool calls, structured outputs, caching, system prompt handling). The rest of the codebase only knows the canonical format. |
| Per-Model Prompt Profiles | Build profiles / configuration variants | Maintain system prompts, few-shot examples, and tool descriptions as versioned variants per model when needed. What works best for Model A can confuse Model B. |
| Native Features over Polyfills | Platform APIs over polyfills | Use native tool calling, structured schemas, and prompt caching rather than emulating them with prompt tricks. Plays to the strengths the model was trained on. |
| Capability Matrix | Feature detection / feature flags | Declare in code what each model supports (`tools`, `vision`, etc.). The router filters candidates accordingly. |
| Graceful Degradation | Progressive enhancement / fallback path | When a model lacks a feature, fall back to a supported alternative (e.g. Pydantic validation + retry) instead of crashing. |
| Cross-Model Evals | Cross-browser testing / benchmark suite | Run the exact same eval suite with real project tasks against every candidate model. Shows which model performs best with this harness and whether a prompt tweak for A regresses B. |
| Task-Based Model Routing | Strategy pattern / capability routing | Pick the best model per task based on evals: fast model for routine steps, frontier model for complex reasoning, specialized model for code or vision. |

> **Related building blocks in this taxonomy:** model version pinning, shadow mode / canary during model migrations, portable tool protocols (MCP), unified logging & tracing for model comparisons, model cascading, fallback, and circuit breakers with provider failover.

## 9. Human–Agent Interface (UX & Transparency)

The user interface is also deterministic code — and often the place that determines whether approvals, corrections, and trust actually work in practice.

| Concept | Classical Equivalent | Purpose (Including Practical Caveats) |
|---|---|---|
| Progress Transparency | Progress bar / activity log | Surface the current step and tool calls via events so humans can intervene early rather than only at the end. |
| Preview Before Execution | Dry-run / diff view | Show planned changes (code diff, email draft, payment details) before executing them. That is what makes human approval meaningful rather than rubber-stamping. |
| Communicating Uncertainty | Warning banner / status indicator | Let the agent ask clarifying questions (`ask_user`) or flag assumptions instead of presenting guesses with false confidence. |
| Undo / Rollback | Undo stack | Provide a clear way for users to revert agent actions (technically backed by compensation logic or version control). |
| Feedback Capture | Rating / bug report | Capture thumbs up/down, edits, and cancellations to feed evals, online monitoring, and the data flywheel. |
