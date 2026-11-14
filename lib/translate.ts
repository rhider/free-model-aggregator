// The translations the starter does not have: every client wire to every provider wire.
//
// The trial's client speaks OpenAI or Anthropic; the providers speak OpenAI, Anthropic or Gemini.
// The starter only forwards OpenAI->OpenAI and translates Anthropic->OpenAI. This file builds the
// other four directions, in JSON and in server-sent events.
//
// The one rule every function here obeys: the provider must echo exactly what the client sent.
// The env's client compares the provider's echo (system sha, user sha, tool names in order,
// max_tokens, stop sequences, image sha, JSON mode) with the request it sent, so a translation
// that rewrites or drops a field loses the request. Nothing here invents text, reorders tools,
// or changes a number.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for every OpenAI-shaped answer this file builds. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of content blocks joined with no separator (an answer's text, as the client reads it). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * System messages join into one system string (the provider echoes their joined sha); tool
 * results merge into the next user message so the echoed user text is unchanged; tool order,
 * max_tokens, stops and the stream flag pass through. response_format has no Anthropic
 * spelling, so the router never sends a JSON-mode request here.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  /** Tool results waiting for the next user message (or the last one, if none follows). */
  const pendingToolResults: Json[] = [];
  const flushToolResults = (blocks: Json[]): void => {
    if (pendingToolResults.length === 0) return;
    blocks.unshift(...pendingToolResults.splice(0, pendingToolResults.length));
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: stringOrEmpty(raw.tool_call_id),
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: stringOrEmpty(call.id),
          name: stringOrEmpty(call.function.name),
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: tool results pending from before it go first, then its own blocks.
    const blocks: Json[] = [];
    flushToolResults(blocks);
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = openAIPartToAnthropic(part);
        if (block !== null) blocks.push(block);
      }
    }
    messages.push({ role: 'user', content: blocks });
  }
  if (pendingToolResults.length > 0) {
    // Tool results with no user message after them: attach them to the last user message, whose
    // echoed text they do not change (only its text blocks count), or stand alone if there is none.
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUser !== undefined && Array.isArray(lastUser.content)) {
      (lastUser.content as Json[]).unshift(...pendingToolResults.splice(0, pendingToolResults.length));
    } else {
      messages.push({ role: 'user', content: pendingToolResults.splice(0, pendingToolResults.length) });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: stringOrEmpty(raw.function.description), input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI message's tool calls, by their id, for naming a tool message's function response. */
function toolCallNames(messages: unknown): Map<string, string> {
  const names = new Map<string, string>();
  if (!Array.isArray(messages)) return names;
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.tool_calls)) continue;
    for (const call of raw.tool_calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
        names.set(call.id, call.function.name);
      }
    }
  }
  return names;
}

/** OpenAI tools as Gemini function declarations, in order. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: stringOrEmpty(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** Anthropic tools as Gemini function declarations, in order. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: stringOrEmpty(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** The generationConfig for a Gemini request: caps, sampling and stops, JSON mode when asked. */
function geminiConfig(body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request naming `model`.
 * The model goes in the URL, not the body. System messages become the systemInstruction;
 * user images become inlineData (same bytes, same echo sha); assistant tool calls become
 * functionCall parts; tool messages become function responses named from the call they answer.
 */
export function openAIToGeminiReq(body: Json, model: string): Json {
  void model;
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const names = toolCallNames(body.messages);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: names.get(stringOrEmpty(raw.tool_call_id)) ?? stringOrEmpty(raw.tool_call_id), response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: stringOrEmpty(call.function.name), args: argumentsOf(call.function.arguments) } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = openAIToolsToGemini(body);
  if (tools.length > 0) out.tools = tools;
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const config = geminiConfig(body, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One Anthropic content block as a Gemini part: text or an inline image; null for the rest. */
function anthropicBlockToGeminiPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** Anthropic tool_use blocks, by their id, for naming a tool result's function response. */
function toolUseNames(messages: unknown): Map<string, string> {
  const names = new Map<string, string>();
  if (!Array.isArray(messages)) return names;
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (isRecord(block) && block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') names.set(block.id, block.name);
    }
  }
  return names;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request naming `model`.
 * The system becomes the systemInstruction; user images become inlineData (same bytes, same
 * echo sha); assistant tool_use blocks become functionCall parts; tool_result blocks become
 * function responses. Anthropic has no JSON mode, so none is set.
 */
export function anthropicToGeminiReq(body: Json, model: string): Json {
  void model;
  const contents: Json[] = [];
  const names = toolUseNames(body.messages);
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: stringOrEmpty(block.name), args: isRecord(block.input) ? block.input : {} } });
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: its tool results go first as a function content, then its text/images.
    const results: Json[] = [];
    const parts: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          const content = block.content;
          results.push({
            functionResponse: {
              name: names.get(stringOrEmpty(block.tool_use_id)) ?? stringOrEmpty(block.tool_use_id),
              response: { content: typeof content === 'string' ? content : textOf(content) },
            },
          });
          continue;
        }
        const part = anthropicBlockToGeminiPart(block);
        if (part !== null) parts.push(part);
      }
    } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
    if (results.length > 0) contents.push({ role: 'function', parts: results });
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = anthropicToolsToGemini(body);
  if (tools.length > 0) out.tools = tools;
  const config = geminiConfig(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer for `model`. */
export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content: Json[] = [];
  const text = answerText(body.content);
  const calls: Json[] = [];
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      calls.push({
        id: stringOrEmpty(block.id),
        type: 'function',
        function: { name: stringOrEmpty(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** The text of a Gemini candidate's parts, joined with no separator. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini answer's finish reason: its first candidate's, or 'STOP' when there is none. */
function geminiFinish(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  return first !== null ? first.finishReason : 'STOP';
}

/** A Gemini answer's usage: prompt and candidate token counts, or zeros. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer for `model`. */
export function geminiToOpenAIResp(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiStopToOpenAI(geminiFinish(body)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages answer for `model`. */
export function geminiToAnthropicResp(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiStopToAnthropic(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One SSE frame of an upstream stream: its event name (null when unnamed) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** An upstream SSE body as frames, in order. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseSseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** One OpenAI Chat Completions chunk object. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** An Anthropic Messages stream as OpenAI Chat Completions chunks, ending with [DONE]. */
export async function* anthropicEventsToOpenAIChunks(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  /** Anthropic content-block index -> OpenAI tool-call index, for tool_use blocks. */
  const toolIndexes = new Map<number, number>();
  let nextToolIndex = 0;
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
    }
    switch (frame.event) {
      case 'message_start': {
        const message = isRecord(data.message) ? data.message : {};
        const usage = isRecord(message.usage) ? message.usage : {};
        inputTokens = numberOrZero(usage.input_tokens);
        break;
      }
      case 'content_block_start': {
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          const index = nextToolIndex;
          nextToolIndex += 1;
          if (typeof data.index === 'number') toolIndexes.set(data.index, index);
          yield `data: ${JSON.stringify(openAIChunk(model, { tool_calls: [{ index, id: stringOrEmpty(block.id), type: 'function', function: { name: stringOrEmpty(block.name), arguments: '' } }] }, null))}\n\n`;
        }
        break;
      }
      case 'content_block_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          yield `data: ${JSON.stringify(openAIChunk(model, { content: delta.text }, null))}\n\n`;
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '' && typeof data.index === 'number') {
          const index = toolIndexes.get(data.index);
          if (index !== undefined) {
            yield `data: ${JSON.stringify(openAIChunk(model, { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null))}\n\n`;
          }
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') pendingStop = anthropicStopToOpenAI(delta.stop_reason);
        const usage = isRecord(data.usage) ? data.usage : {};
        outputTokens = numberOrZero(usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  if (!started) yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  if (!finished) {
    finished = true;
    yield `data: ${JSON.stringify(openAIChunk(model, {}, pendingStop ?? 'stop'))}\n\n`;
    yield `data: ${JSON.stringify({ id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}

/**
 * A Gemini streamGenerateContent stream as OpenAI Chat Completions chunks, ending with [DONE].
 * Gemini frames carry text pieces; the last frame carries the finish reason and usageMetadata.
 */
export async function* geminiFramesToOpenAIChunks(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
    }
    const text = geminiText(data);
    if (text !== '') yield `data: ${JSON.stringify(openAIChunk(model, { content: text }, null))}\n\n`;
    const finish = geminiFinish(data);
    if (typeof finish === 'string' && finish !== '') pendingStop = geminiStopToOpenAI(finish);
    const usage = geminiUsage(data);
    if (usage.input !== 0 || usage.output !== 0) {
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
  }
  if (!started) yield `data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`;
  if (!finished) {
    finished = true;
    yield `data: ${JSON.stringify(openAIChunk(model, {}, pendingStop ?? 'stop'))}\n\n`;
    yield `data: ${JSON.stringify({ id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}

/**
 * A Gemini streamGenerateContent stream as Anthropic Messages events: message_start, one text
 * block, its deltas, then message_delta with the stop reason and output tokens, message_stop.
 * The input token count is only known at the last frame, and the client's usage check on an
 * Anthropic stream compares output tokens alone, so message_start carries zeros.
 */
export async function* geminiFramesToAnthropicEvents(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let blockOpen = false;
  let finished = false;
  let pendingStop = 'end_turn';
  let outputTokens = 0;
  for await (const frame of sseFrames(stream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (!started) {
      started = true;
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: `msg_${model}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
    }
    const text = geminiText(data);
    if (text !== '') {
      if (!blockOpen) {
        blockOpen = true;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    }
    const finish = geminiFinish(data);
    if (typeof finish === 'string' && finish !== '') pendingStop = geminiStopToAnthropic(finish);
    const usage = geminiUsage(data);
    if (usage.output !== 0) outputTokens = usage.output;
  }
  if (!started) {
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: `msg_${model}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    });
    started = true;
  }
  if (!finished) {
    finished = true;
    if (blockOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop, stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}
