// Run after pnpm build: node tests/smoke_runtime.mjs <runtime-directory> <model.gguf>
import assert from 'node:assert/strict';
import net from 'node:net';
import { LlamaRuntime } from '../packages/runtime-llama/dist/index.js';

const [runtimeDir, modelPath] = process.argv.slice(2);
assert.ok(runtimeDir && modelPath, 'Supply an installed runtime directory and a local GGUF model. Nothing is downloaded.');
const listener = net.createServer();
await new Promise(resolve => listener.listen(0, '127.0.0.1', resolve));
const port = listener.address().port;
await new Promise(resolve => listener.close(resolve));
const runtime = new LlamaRuntime(runtimeDir);
try {
  await runtime.start({
    instanceId: 'live-smoke', modelPath, modelName: 'live-smoke', runtimeDir,
    port, host: '127.0.0.1', contextSize: 512, gpuLayers: 0, threads: 1,
    flashAttention: false,
  });
  assert.equal((await runtime.health()).modelLoaded, true);
  let content = '';
  for await (const chunk of runtime.chatStream({
    model: 'default', messages: [{ role: 'user', content: 'What is the capital of France?' }],
    max_tokens: 24, temperature: 0,
  }, AbortSignal.timeout(30000))) {
    content += chunk.choices[0]?.delta?.content || '';
  }
  assert.ok(content.trim(), 'The healthy runtime must actually generate text.');
  assert.equal((await runtime.detectCapabilities()).supportsEmbeddings, false);
  await assert.rejects(runtime.embeddings({ model: 'default', input: 'Hello' }), {
    code: 'EMBEDDINGS_UNAVAILABLE', statusCode: 503,
  });
  console.log(JSON.stringify({ chat: content, embeddings: 'explicitly unavailable', backend: 'cpu', threads: 1 }));
} finally {
  await runtime.stop();
}
