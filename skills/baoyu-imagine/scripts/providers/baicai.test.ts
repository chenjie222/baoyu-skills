import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";

import type { CliArgs } from "../types.ts";
import {
  buildRequestBody,
  detectTaskType,
  extractImmediateUrl,
  extractTaskId,
  extractUrlFromResultBody,
  generateImage,
  getBaseUrl,
  getDefaultModel,
  getModelFamily,
  pollTaskUntilComplete,
  resolveCommercialResolution,
  resolveOpenSourceSize,
  uploadReferenceImage,
  validateArgs,
} from "./baicai.ts";

function makeArgs(overrides: Partial<CliArgs> = {}): CliArgs {
  return {
    prompt: null,
    promptFiles: [],
    imagePath: null,
    provider: null,
    model: null,
    aspectRatio: null,
    size: null,
    quality: null,
    imageSize: null,
    imageApiDialect: null,
    referenceImages: [],
    n: 1,
    batchFile: null,
    jobs: null,
    json: false,
    help: false,
    ...overrides,
  };
}

function useEnv(t: TestContext, values: Record<string, string | null>): void {
  const previous = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(values)) {
    previous.set(key, process.env[key]);
    if (value == null) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  t.after(() => {
    for (const [key, value] of previous.entries()) {
      if (value == null) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  });
}

type FetchCall = { url: string; init: RequestInit | undefined };

function stubFetch(
  t: TestContext,
  responder: (call: FetchCall, index: number) => Response | Promise<Response>,
): { calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : (input as Request).url;
    const call: FetchCall = { url, init };
    calls.push(call);
    return Promise.resolve(responder(call, calls.length - 1));
  };
  t.after(() => {
    globalThis.fetch = original;
  });
  return { calls };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function bytesResponse(bytes: number[]): Response {
  return new Response(Uint8Array.from(bytes), {
    status: 200,
    headers: { "Content-Type": "image/png" },
  });
}

test("baicai getDefaultModel honors env override and falls back to FLUX.1-dev", (t) => {
  useEnv(t, { BAICAI_IMAGE_MODEL: null });
  assert.equal(getDefaultModel(), "FLUX.1-dev");
  process.env.BAICAI_IMAGE_MODEL = "Wan2.5-Image";
  assert.equal(getDefaultModel(), "Wan2.5-Image");
});

test("baicai getBaseUrl trims trailing slash and respects override", (t) => {
  useEnv(t, { BAICAI_BASE_URL: null });
  assert.equal(getBaseUrl(), "https://cloud.baicaiinfer.com");
  process.env.BAICAI_BASE_URL = "https://proxy.example.com/";
  assert.equal(getBaseUrl(), "https://proxy.example.com");
});

test("baicai model family: FLUX open-source, Wan commercial, unknown commercial", () => {
  assert.equal(getModelFamily("FLUX.1-dev"), "open-source");
  assert.equal(getModelFamily("Qwen-Image-Edit"), "open-source");
  assert.equal(getModelFamily("Wan2.5-Image"), "commercial");
  assert.equal(getModelFamily("brand-new-model"), "commercial");
});

test("baicai detectTaskType covers the truth table", () => {
  assert.equal(detectTaskType("FLUX.1-dev", makeArgs()), "txt2img");
  assert.equal(
    detectTaskType("FLUX.1-Kontext-dev", makeArgs({ referenceImages: ["ref.png"] })),
    "image-edit",
  );
  assert.equal(
    detectTaskType("Qwen-Image-Edit-2509", makeArgs({ referenceImages: ["ref.png"] })),
    "image-edit",
  );
  assert.equal(
    detectTaskType("FLUX.1-dev", makeArgs({ referenceImages: ["ref.png"] })),
    "img2img",
  );
  assert.equal(
    detectTaskType("Wan2.5-Image", makeArgs({ referenceImages: ["ref.png"] })),
    "img2img",
  );
  assert.throws(
    () => detectTaskType("FLUX.1-dev", makeArgs({ referenceImages: ["a.png", "b.png"] })),
    /at most 1 reference image/,
  );
});

test("baicai resolveOpenSourceSize handles defaults, --size, and --ar", () => {
  assert.deepEqual(resolveOpenSourceSize({ size: null, aspectRatio: null, quality: null }), {
    width: 1024,
    height: 1024,
  });
  assert.deepEqual(resolveOpenSourceSize({ size: "1280x768", aspectRatio: null, quality: null }), {
    width: 1280,
    height: 768,
  });
  const ar = resolveOpenSourceSize({ size: null, aspectRatio: "16:9", quality: "2k" });
  assert.ok(ar.width % 64 === 0 && ar.height % 64 === 0);
  assert.ok(ar.width > ar.height);
  assert.ok(ar.width <= 2048 && ar.height >= 512);
  assert.throws(
    () => resolveOpenSourceSize({ size: "huge", aspectRatio: null, quality: null }),
    /WxH format/,
  );
});

test("baicai resolveCommercialResolution respects imageSize then quality", () => {
  assert.equal(resolveCommercialResolution({ imageSize: "4K", quality: null }), "4k");
  assert.equal(resolveCommercialResolution({ imageSize: "2K", quality: null }), "2k");
  assert.equal(resolveCommercialResolution({ imageSize: null, quality: "2k" }), "2k");
  assert.equal(resolveCommercialResolution({ imageSize: null, quality: "normal" }), "1k");
  assert.equal(resolveCommercialResolution({ imageSize: null, quality: null }), "1k");
});

test("baicai validateArgs rejects n>1, bad commercial --ar, and commercial --size", () => {
  assert.throws(
    () => validateArgs("FLUX.1-dev", makeArgs({ n: 2 })),
    /single image per request/,
  );
  assert.throws(
    () => validateArgs("Wan2.5-Image", makeArgs({ aspectRatio: "5:4" })),
    /commercial models only accept --ar/,
  );
  assert.throws(
    () => validateArgs("Wan2.5-Image", makeArgs({ size: "1280x720" })),
    /do not accept --size/,
  );
  validateArgs("FLUX.1-dev", makeArgs({ size: "1280x768" }));
  validateArgs("Wan2.5-Image", makeArgs({ aspectRatio: "16:9" }));
});

test("baicai buildRequestBody: open-source txt2img / img2img / defaults", () => {
  const txt2img = buildRequestBody("hello", "FLUX.1-dev", makeArgs(), null);
  assert.equal(txt2img.selected_model, "FLUX.1-dev");
  assert.equal(txt2img.task_type, "txt2img");
  assert.equal(txt2img.response_format, "url");
  assert.deepEqual(txt2img.input, {
    prompt: "hello",
    number_of_images: 1,
    width: 1024,
    height: 1024,
  });

  const edit = buildRequestBody(
    "studio ghibli style",
    "FLUX.1-Kontext-dev",
    makeArgs({ referenceImages: ["ref.png"], size: "1024x1024" }),
    "https://uploaded/ref.png",
  );
  assert.equal(edit.task_type, "image-edit");
  assert.equal((edit.input as { image?: string }).image, "https://uploaded/ref.png");
});

test("baicai buildRequestBody: commercial txt2img maps resolution + aspect_ratio", () => {
  const body = buildRequestBody(
    "a cinematic skyline",
    "Wan2.5-Image",
    makeArgs({ aspectRatio: "16:9", quality: "2k" }),
    null,
  );
  assert.equal(body.task_type, "txt2img");
  assert.deepEqual(body.input, {
    prompt: "a cinematic skyline",
    number_of_images: 1,
    resolution: "2k",
    aspect_ratio: "16:9",
  });
});

test("baicai extract helpers branch on response shape", () => {
  assert.equal(extractImmediateUrl({ data: { url: ["https://x/a.png"] } }), "https://x/a.png");
  assert.equal(extractImmediateUrl({ data: [{ taskId: "t1" }] }), null);
  assert.equal(extractTaskId({ data: [{ taskId: "t1" }] }), "t1");
  assert.equal(extractTaskId({ data: { url: ["https://x"] } } as unknown as { data?: never }), null);
  assert.equal(
    extractUrlFromResultBody({ data: { urls: ["https://x/r.png"] } }),
    "https://x/r.png",
  );
  assert.equal(
    extractUrlFromResultBody({
      data: [{ result: { urls: ["https://x/r2.png"] } }],
    }),
    "https://x/r2.png",
  );
  assert.equal(extractUrlFromResultBody({}), null);
});

test("baicai FLUX happy path: generate via immediate URL", async (t) => {
  useEnv(t, { BAICAI_API_KEY: "sk-test", BAICAI_BASE_URL: "https://cloud.baicaiinfer.com" });
  const { calls } = stubFetch(t, (call) => {
    if (call.url.endsWith("/v1/images/generations")) {
      return jsonResponse({ code: 0, data: { url: ["https://cdn.example.com/out.png"] } });
    }
    if (call.url === "https://cdn.example.com/out.png") {
      return bytesResponse([10, 20, 30]);
    }
    throw new Error(`unexpected url: ${call.url}`);
  });

  const bytes = await generateImage("a red fox", "FLUX.1-dev", makeArgs());
  assert.deepEqual([...bytes], [10, 20, 30]);
  assert.equal(calls.length, 2);
  const body = JSON.parse((calls[0]!.init?.body as string) ?? "{}");
  assert.equal(body.selected_model, "FLUX.1-dev");
  assert.equal(body.task_type, "txt2img");
});

test("baicai commercial polling happy path: PENDING then COMPLETED", async (t) => {
  useEnv(t, { BAICAI_API_KEY: "sk-test", BAICAI_BASE_URL: "https://cloud.baicaiinfer.com" });
  const statuses = ["PENDING", "RUNNING", "COMPLETED"];
  let statusIndex = 0;
  const { calls } = stubFetch(t, (call) => {
    if (call.url.endsWith("/v1/images/generations")) {
      return jsonResponse({ code: 0, data: [{ taskId: "task-42", status: "PENDING" }] });
    }
    if (call.url.includes("/v1/comfyui/tasks/task-42/status")) {
      const status = statuses[statusIndex++] ?? "COMPLETED";
      return jsonResponse({ code: 0, data: [{ status }] });
    }
    if (call.url.endsWith("/v1/comfyui/tasks/task-42/result")) {
      return jsonResponse({ code: 0, data: { urls: ["https://cdn.example.com/out.png"] } });
    }
    if (call.url === "https://cdn.example.com/out.png") {
      return bytesResponse([1, 2, 3, 4]);
    }
    throw new Error(`unexpected url: ${call.url}`);
  });

  const bytes = await generateImage(
    "city skyline",
    "Wan2.5-Image",
    makeArgs({ aspectRatio: "16:9", quality: "2k" }),
    { sleep: async () => undefined },
  );
  assert.deepEqual([...bytes], [1, 2, 3, 4]);
  assert.ok(calls.length >= 5);
});

test("baicai polling fails fast on FAILED status", async (t) => {
  stubFetch(t, () => jsonResponse({ code: 0, data: [{ status: "FAILED" }] }));
  await assert.rejects(
    pollTaskUntilComplete("t-err", "sk", "https://cloud.baicaiinfer.com", { sleep: async () => undefined }),
    /ended with status FAILED/,
  );
});

test("baicai polling times out after max attempts", async (t) => {
  stubFetch(t, () => jsonResponse({ code: 0, data: [{ status: "RUNNING" }] }));
  await assert.rejects(
    pollTaskUntilComplete("t-slow", "sk", "https://cloud.baicaiinfer.com", { sleep: async () => undefined }),
    /polling timed out/,
  );
});

test("baicai uploadReferenceImage: HTTPS passthrough and local multipart upload", async (t) => {
  const httpsUrl = "https://cdn.example.com/ref.png";
  assert.equal(
    await uploadReferenceImage(httpsUrl, "sk", "https://cloud.baicaiinfer.com"),
    httpsUrl,
  );

  const tmp = mkdtempSync(path.join(os.tmpdir(), "baicai-"));
  const local = path.join(tmp, "hello.png");
  writeFileSync(local, new Uint8Array([0xff, 0xd8, 0xff, 0xe0]));
  t.after(() => rmSync(tmp, { recursive: true, force: true }));

  const { calls } = stubFetch(t, () =>
    jsonResponse({ code: 0, data: { fullPath: "https://cdn.example.com/uploaded/hello.png" } }),
  );
  const url = await uploadReferenceImage(local, "sk", "https://cloud.baicaiinfer.com");
  assert.equal(url, "https://cdn.example.com/uploaded/hello.png");
  assert.equal(calls.length, 1);
  assert.match(calls[0]!.url, /\/v1\/resources\/upload$/);
});

test("baicai generateImage throws when BAICAI_API_KEY is missing", async (t) => {
  useEnv(t, { BAICAI_API_KEY: null });
  await assert.rejects(
    generateImage("x", "FLUX.1-dev", makeArgs()),
    /BAICAI_API_KEY is required/,
  );
});

test("baicai generateImage rejects >1 reference images via validateArgs", async (t) => {
  useEnv(t, { BAICAI_API_KEY: "sk-test" });
  await assert.rejects(
    generateImage(
      "x",
      "FLUX.1-Kontext-dev",
      makeArgs({ referenceImages: ["a.png", "b.png"] }),
    ),
    /at most 1 reference image/,
  );
});
