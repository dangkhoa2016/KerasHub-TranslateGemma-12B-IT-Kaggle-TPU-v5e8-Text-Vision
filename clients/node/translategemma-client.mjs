#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

function option(args, name, fallback = undefined) {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
}

function requireValue(value, message) {
  if (!value) throw new Error(message);
  return value;
}

class Client {
  constructor({ baseUrl, apiKey, requestId, pollInterval = 2000, pollTimeout = 1800000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.requestId = requestId;
    this.pollInterval = pollInterval;
    this.pollTimeout = pollTimeout;
  }

  headers(extra = {}) {
    const headers = { ...extra };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    if (this.requestId) headers['X-Request-ID'] = this.requestId;
    return headers;
  }

  async request(method, pathname, { jsonBody, body, headers = {} } = {}) {
    const init = { method, headers: this.headers(headers) };
    if (jsonBody !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(jsonBody);
    } else if (body !== undefined) {
      init.body = body;
    }
    const response = await fetch(this.baseUrl + pathname, init);
    const text = await response.text();
    let payload = {};
    if (text) {
      try { payload = JSON.parse(text); } catch { payload = { error: text }; }
    }
    return [response.status, payload];
  }

  async submitAndWait(pathname, requestOptions) {
    let [status, payload] = await this.request('POST', pathname, requestOptions);
    if (![200, 202].includes(status)) throw new Error(`HTTP ${status}: ${payload.error ?? JSON.stringify(payload)}`);
    if (status === 200) return payload;
    const resultUrl = payload.result_url ?? (payload.job_id ? `/result/${payload.job_id}` : null);
    if (!resultUrl) throw new Error('202 response has no result URL');
    const deadline = Date.now() + this.pollTimeout;
    while (Date.now() < deadline) {
      [status, payload] = await this.request('GET', resultUrl);
      if (status === 200) return payload;
      if (status !== 202) throw new Error(`HTTP ${status}: ${payload.error ?? JSON.stringify(payload)}`);
      await new Promise(resolve => setTimeout(resolve, this.pollInterval));
    }
    throw new Error(`Timed out waiting for ${resultUrl}`);
  }

  translateText(text, sourceLang, targetLang, maxNewTokens = 256) {
    return this.submitAndWait('/translate', {
      jsonBody: { text, source_lang: sourceLang, target_lang: targetLang, max_new_tokens: maxNewTokens },
    });
  }

  async translateImage(imagePath, sourceLang, targetLang, maxNewTokens = 256) {
    const bytes = fs.readFileSync(imagePath);
    const form = new FormData();
    form.append('source_lang', sourceLang);
    form.append('target_lang', targetLang);
    form.append('max_new_tokens', String(maxNewTokens));
    form.append('image', new Blob([bytes]), path.basename(imagePath));
    return this.submitAndWait('/translate/image', { body: form });
  }

  async info() {
    const [status, payload] = await this.request('GET', '/info');
    if (status !== 200) throw new Error(`HTTP ${status}: ${payload.error ?? JSON.stringify(payload)}`);
    return payload;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const command = requireValue(args[0], 'Usage: translategemma-client.mjs <text|image|info> [options]');
  const baseUrl = option(args, '--base-url', 'http://127.0.0.1:7860');
  const apiKeyFile = option(args, '--api-key-file');
  const apiKey = option(args, '--api-key') ?? (apiKeyFile ? fs.readFileSync(apiKeyFile, 'utf8').trim() : undefined);
  const client = new Client({ baseUrl, apiKey, requestId: option(args, '--request-id') });
  let result;
  if (command === 'text') {
    const value = requireValue(option(args, '--text'), '--text is required');
    result = await client.translateText(value, option(args, '--source-lang', 'English'), option(args, '--target-lang', 'Vietnamese'), Number(option(args, '--max-new-tokens', '256')));
  } else if (command === 'image') {
    const value = requireValue(option(args, '--image'), '--image is required');
    result = await client.translateImage(value, option(args, '--source-lang', 'English'), option(args, '--target-lang', 'Vietnamese'), Number(option(args, '--max-new-tokens', '256')));
  } else if (command === 'info') {
    result = await client.info();
  } else {
    throw new Error(`Unknown command: ${command}`);
  }
  console.log(JSON.stringify(result, null, 2));
}

main().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
