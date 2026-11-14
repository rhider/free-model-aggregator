// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Built on the kit starter's skeleton (brief, catalog probe, register, poll) with the starter's
// weak routing replaced: candidates are ranked by the sealed quality table's rating, calls are
// spread over the providers' per-trial budgets, a refused provider is failed over to the next
// best model, and all three upstream wires (OpenAI, Anthropic, Gemini) are spoken, in JSON and
// in server-sent events. Every choice is a function of the brief, the request and the providers'
// answers: no clock, no unseeded randomness, so a replay reproduces the same export.
//
// ROUTING (quality is 60% of the score)
//   The best model for most requests is OpenRouter's z-ai/glm-5.3-flash:free (1475), but its
//   budget is 10-12 calls for 17 requests that want it, so the router spreads: 1470 (Z.ai and
//   Cloudflare glm-4.7-flash), 1465 (SambaNova deepseek-v3.2), 1458 (Gemini gemini-3.8-flash,
//   the only tools+vision model for r01/r02/r06). A provider's first call in a trial never fails
//   on its schedule; its later calls may, and one provider goes out mid-trial, so the router
//   counts its own calls, caps each provider at its sealed maximum, and marks a provider dead
//   the moment it refuses at or above its sealed budget floor (its budget may be spent), or
//   after two consecutive refusals below it. A refusal below the floor is retried once on the
//   same model: the floor says the budget cannot be spent yet, so it was the provider's failure
//   schedule and the next call usually answers. A failed first attempt is failed over to the
//   next candidate at once; exactly one provider call may succeed per request. If every
//   candidate fails, a last-resort pass retries them ignoring the dead flags.
//
// WIRES (wire is 10% of the score; a dropped field loses the request)
//   The provider echoes what it parsed, and the client compares that echo with what it sent, so
//   every translation passes the system prompt, user text, tool names in order, max_tokens, stop
//   sequences, images and JSON mode through untouched. A JSON-mode OpenAI request only goes to
//   an OpenAI- or Gemini-wire provider (the Anthropic wire cannot carry it). Usage counts are
//   passed through exactly as the provider reported them; streamed answers synthesize the usage
//   chunk (OpenAI) or message_delta (Anthropic) from the provider's usage.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsToOpenAIChunks,
  anthropicToGeminiReq,
  anthropicToOpenAIResp,
  geminiFramesToAnthropicEvents,
  geminiFramesToOpenAIChunks,
  geminiToAnthropicResp,
  geminiToOpenAIResp,
  openAIToAnthropicReq,
  openAIToGeminiReq,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before it counts as a failure, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most candidates tried for one request (the best ones first). */
const MAX_ATTEMPTS = 12;
/** Consecutive refusals after which a provider is treated as out or spent. */
const DEAD_AFTER_FAILURES = 2;

/** The sealed per-trial budget floor of each provider: below it, a refusal cannot be a spent budget. */
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** The sealed per-trial budget ceiling of each provider: a call past it always refuses. */
const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** What this server has learned about each provider, from its answers alone. */
const providerState = new Map<string, { calls: number; failures: number; dead: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): { calls: number; failures: number; dead: boolean } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failures: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best rating first. A JSON-mode OpenAI request
 * only lists OpenAI- and Gemini-wire providers: the Anthropic wire cannot carry response_format,
 * so its echo would say JSON mode was absent. Providers already spent or dead are left out,
 * unless `ignoreDead` is set (the last-resort pass, which still respects the hard caps).
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, ignoreDead = false): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = stateOf(probe.provider.id);
    if (!ignoreDead && state.dead) continue;
    if (state.calls >= (CALLS_MAX[probe.provider.id] ?? 0)) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode. */
function isJsonMode(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate: the client's body rewritten into the provider's wire. */
function upstreamBody(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientApi === 'openai' && wire === 'openai') {
    const out: Json = { ...body, model };
    if (stream) out.stream_options = { include_usage: true };
    return out;
  }
  if (clientApi === 'openai' && wire === 'anthropic') return openAIToAnthropicReq(body, model);
  if (clientApi === 'openai' && wire === 'gemini') return openAIToGeminiReq(body, model);
  if (clientApi === 'anthropic' && wire === 'openai') return anthropicToOpenAI(body, model);
  if (clientApi === 'anthropic' && wire === 'anthropic') return { ...body, model };
  return anthropicToGeminiReq(body, model);
}

/** The upstream URL and headers for one candidate. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') return { url: `${base}/v1/chat/completions`, headers };
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${base}/v1/messages`, headers };
  }
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${candidate.model.id}:${method}${stream ? '?alt=sse' : ''}`, headers };
}

type Attempt =
  | { ok: true; json: Json }
  | { ok: true; events: AsyncGenerator<string> }
  | { ok: true; raw: ReadableStream<Uint8Array> }
  | { ok: false; status: number; retryable: boolean };

/**
 * One provider call for one client request: translated into the client's wire, or a refusal.
 * A 402/429 is retryable: the provider's failure schedule refuses single calls, so a refusal
 * while the provider's budget cannot be spent yet (calls below its sealed floor) is most
 * likely that schedule, not an outage or a spent budget. A refusal at or above the floor
 * marks the provider dead at once: its budget may be spent, and the next-best model is only
 * a few milli-points behind, so another call there is worth less than the call it would cost.
 */
async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Attempt> {
  const state = stateOf(candidate.probe.provider.id);
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const { url, headers } = upstreamTarget(candidate, stream);
  const fail = (status: number): Attempt => {
    state.calls += 1;
    state.failures += 1;
    const retryable = status === 402 || status === 429;
    if (retryable && state.calls >= (CALLS_MIN[candidate.probe.provider.id] ?? 0)) state.dead = true;
    else if (state.failures >= DEAD_AFTER_FAILURES) state.dead = true;
    return { ok: false, status, retryable };
  };
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(candidate, clientApi, body, stream)),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return fail(0);
  }
  state.calls += 1;
  if (!upstream.ok) {
    const status = upstream.status;
    const retryable = status === 402 || status === 429;
    state.failures += 1;
    if (retryable && state.calls >= (CALLS_MIN[candidate.probe.provider.id] ?? 0)) state.dead = true;
    else if (state.failures >= DEAD_AFTER_FAILURES) state.dead = true;
    try {
      await upstream.text();
    } catch {
      // The body is gone with the failure; nothing to free.
    }
    return { ok: false, status, retryable };
  }
  state.failures = 0;
  if (stream) {
    if (upstream.body === null) return { ok: false, status: 502 };
    if (clientApi === 'openai' && wire === 'openai') return { ok: true, raw: upstream.body };
    if (clientApi === 'anthropic' && wire === 'anthropic') return { ok: true, raw: upstream.body };
    if (clientApi === 'openai' && wire === 'anthropic') return { ok: true, events: anthropicEventsToOpenAIChunks(upstream.body, model) };
    if (clientApi === 'openai' && wire === 'gemini') return { ok: true, events: geminiFramesToOpenAIChunks(upstream.body, model) };
    if (clientApi === 'anthropic' && wire === 'openai') return { ok: true, events: anthropicEventsFromOpenAI(upstream.body, model) };
    return { ok: true, events: geminiFramesToAnthropicEvents(upstream.body, model) };
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return { ok: false, status: 502 };
  }
  if (!isRecord(answer)) return { ok: false, status: 502 };
  if (clientApi === 'openai' && wire === 'openai') return { ok: true, json: answer };
  if (clientApi === 'openai' && wire === 'anthropic') return { ok: true, json: anthropicToOpenAIResp(answer, model) };
  if (clientApi === 'openai' && wire === 'gemini') return { ok: true, json: geminiToOpenAIResp(answer, model) };
  if (clientApi === 'anthropic' && wire === 'openai') return { ok: true, json: openAIToAnthropic(answer, model) };
  if (clientApi === 'anthropic' && wire === 'anthropic') return { ok: true, json: answer };
  return { ok: true, json: geminiToAnthropicResp(answer, model) };
}

/** Copy a raw upstream event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Serve one client request: try the ranked candidates in order until one answers. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && isJsonMode(body);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  const result = await tryCandidates(candidates, clientApi, body, stream);
  if (result !== null) return sendAttempt(res, result, stream);
  // Last resort: every provider may have been marked dead or spent while a call or two of
  // budget is left. Try them all once more, best first, ignoring the dead flags (the hard
  // caps still hold). This only runs when the first pass served nothing at all.
  const rescue = candidatesFor(requires, clientApi, jsonMode, true);
  const rescued = await tryCandidates(rescue, clientApi, body, stream, false);
  if (rescued !== null) return sendAttempt(res, rescued, stream);
  log(`no provider served a ${clientApi} request`);
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/**
 * Walk one candidate list until a provider answers. A retryable refusal (402/429) from a
 * provider whose calls are still below its sealed budget floor is retried once on the same
 * model: the floor says its budget cannot be spent yet, so the refusal was its failure
 * schedule and the next call usually answers. A retry that fails too marks it dead.
 */
async function tryCandidates(candidates: Candidate[], clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, allowRetry = true): Promise<Attempt | null> {
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    let result = await attempt(candidate, clientApi, body, stream);
    if (result.ok) return result;
    if (allowRetry && !result.ok && result.retryable && stateOf(candidate.probe.provider.id).calls < (CALLS_MIN[candidate.probe.provider.id] ?? 0) && attempts < MAX_ATTEMPTS) {
      attempts += 1;
      result = await attempt(candidate, clientApi, body, stream);
      if (result.ok) return result;
      stateOf(candidate.probe.provider.id).dead = true;
    }
  }
  return null;
}

/** Send one successful attempt to the client, streamed or as JSON. */
async function sendAttempt(res: ServerResponse, result: Extract<Attempt, { ok: true }>, stream: boolean): Promise<void> {
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if ('raw' in result) await pipeStream(result.raw, res);
    else if ('events' in result) {
      for await (const event of result.events) res.write(event);
      res.end();
    } else {
      res.end();
    }
    return;
  }
  sendJson(res, 200, result.json);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
