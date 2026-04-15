import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

import type { CliArgs } from "../types";

export const DEFAULT_MODEL = "FLUX.1-dev";
export const DEFAULT_BASE_URL = "https://cloud.baicaiinfer.com";
export const POLL_INTERVAL_MS = 10_000;
export const POLL_MAX_ATTEMPTS = 120;

const OPEN_SOURCE_MODELS = new Set<string>([
  "FLUX.1-dev",
  "FLUX.2-dev",
  "FLUX.1-schnell",
  "FLUX.1-Kontext-dev",
  "HiDream-I1-Full",
  "HiDream-I1-Dev",
  "HiDream-I1-Fast",
  "Qwen-Image",
  "Qwen-Image-Edit",
  "Qwen-Image-Edit-2509",
  "Kolors",
  "HunyuanDiT",
  "SD3.5-Large",
  "stable-diffusion-v1-5",
  "Z-Image-Turbo",
]);

const EDIT_CAPABLE_MODELS = new Set<string>([
  "FLUX.1-Kontext-dev",
  "Qwen-Image-Edit",
  "Qwen-Image-Edit-2509",
]);

const COMMERCIAL_AR_ALLOWED = new Set<string>([
  "1:1",
  "16:9",
  "9:16",
  "4:3",
  "3:4",
  "3:2",
  "2:3",
  "21:9",
  "9:21",
]);

const OPEN_SOURCE_SIZE_STEP = 64;
const OPEN_SOURCE_MIN_EDGE = 512;
const OPEN_SOURCE_MAX_EDGE = 2048;

export type BaicaiModelFamily = "open-source" | "commercial";
export type BaicaiTaskType = "txt2img" | "img2img" | "image-edit";
export type BaicaiResolution = "1k" | "2k" | "4k";

type OpenSourceInput = {
  prompt: string;
  number_of_images: number;
  width: number;
  height: number;
  image?: string;
};

type CommercialInput = {
  prompt: string;
  number_of_images: number;
  resolution: BaicaiResolution;
  aspect_ratio: string;
  image?: string;
};

export type BaicaiRequestBody = {
  selected_model: string;
  task_type: BaicaiTaskType;
  response_format: "url";
  input: OpenSourceInput | CommercialInput;
};

type BaicaiImmediateResponse = {
  code?: number;
  message?: string;
  data?: {
    url?: string[];
  };
};

type BaicaiTaskResponse = {
  code?: number;
  message?: string;
  data?: Array<{
    taskId?: string;
    status?: string;
  }>;
};

type BaicaiStatusResponse = {
  code?: number;
  data?: Array<{
    status?: string;
    progress?: string;
  }>;
};

type BaicaiResultResponse = {
  code?: number;
  data?:
    | {
        urls?: string[];
        result?: { urls?: string[] };
      }
    | Array<{
        result?: { urls?: string[] };
        urls?: string[];
      }>;
};

type BaicaiUploadResponse = {
  code?: number;
  message?: string;
  data?: {
    fullPath?: string;
  };
};

export type PollDeps = {
  sleep: (ms: number) => Promise<void>;
};

const defaultPollDeps: PollDeps = {
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

export function getDefaultModel(): string {
  return process.env.BAICAI_IMAGE_MODEL || DEFAULT_MODEL;
}

export function getApiKey(): string | null {
  return process.env.BAICAI_API_KEY || null;
}

export function getBaseUrl(): string {
  const base = (process.env.BAICAI_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/g, "");
  return base;
}

export function getModelFamily(model: string): BaicaiModelFamily {
  return OPEN_SOURCE_MODELS.has(model) ? "open-source" : "commercial";
}

export function detectTaskType(model: string, args: CliArgs): BaicaiTaskType {
  const refCount = args.referenceImages.length;
  if (refCount > 1) {
    throw new Error(
      "baicai v1 supports at most 1 reference image (img2img / image-edit). Multi-reference workflows are not yet supported.",
    );
  }
  if (refCount === 1) {
    return EDIT_CAPABLE_MODELS.has(model) ? "image-edit" : "img2img";
  }
  return "txt2img";
}

export function parseAspectRatio(ar: string): { width: number; height: number } | null {
  const match = ar.match(/^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/);
  if (!match) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

export function parseSize(size: string): { width: number; height: number } | null {
  const match = size.trim().match(/^(\d+)\s*[xX*]\s*(\d+)$/);
  if (!match) return null;
  const width = parseInt(match[1]!, 10);
  const height = parseInt(match[2]!, 10);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

function roundToStep(value: number, step: number): number {
  return Math.max(step, Math.round(value / step) * step);
}

function clampEdge(value: number): number {
  return Math.min(Math.max(value, OPEN_SOURCE_MIN_EDGE), OPEN_SOURCE_MAX_EDGE);
}

function getOpenSourceTargetPixels(quality: CliArgs["quality"]): number {
  return quality === "normal" ? 1024 * 1024 : 1536 * 1536;
}

export function resolveOpenSourceSize(
  args: Pick<CliArgs, "size" | "aspectRatio" | "quality">,
): { width: number; height: number } {
  if (args.size) {
    const parsed = parseSize(args.size);
    if (!parsed) {
      throw new Error("baicai --size must be in WxH format, for example 1024x1024.");
    }
    const width = roundToStep(clampEdge(parsed.width), OPEN_SOURCE_SIZE_STEP);
    const height = roundToStep(clampEdge(parsed.height), OPEN_SOURCE_SIZE_STEP);
    return { width, height };
  }

  if (args.aspectRatio) {
    const ratio = parseAspectRatio(args.aspectRatio);
    if (!ratio) {
      throw new Error(`baicai --ar must be in W:H format, got ${args.aspectRatio}.`);
    }
    const target = getOpenSourceTargetPixels(args.quality);
    const scale = Math.sqrt(target / (ratio.width * ratio.height));
    const width = roundToStep(clampEdge(ratio.width * scale), OPEN_SOURCE_SIZE_STEP);
    const height = roundToStep(clampEdge(ratio.height * scale), OPEN_SOURCE_SIZE_STEP);
    return { width, height };
  }

  return { width: 1024, height: 1024 };
}

export function resolveCommercialResolution(
  args: Pick<CliArgs, "quality" | "imageSize">,
): BaicaiResolution {
  if (args.imageSize === "4K") return "4k";
  if (args.imageSize === "2K") return "2k";
  if (args.imageSize === "1K") return "1k";
  if (args.quality === "2k") return "2k";
  return "1k";
}

export function validateArgs(model: string, args: CliArgs): void {
  if (args.n > 1) {
    throw new Error(
      "baicai image generation currently returns a single image per request in baoyu-imagine. Run multiple tasks instead of setting --n > 1.",
    );
  }

  const family = getModelFamily(model);
  if (family === "commercial") {
    if (args.aspectRatio && !COMMERCIAL_AR_ALLOWED.has(args.aspectRatio)) {
      throw new Error(
        `baicai commercial models only accept --ar in ${[...COMMERCIAL_AR_ALLOWED].join(", ")}. Got ${args.aspectRatio}.`,
      );
    }
    if (args.size) {
      throw new Error(
        "baicai commercial models do not accept --size. Use --ar and --quality (or --imageSize) instead.",
      );
    }
  }

  if (args.referenceImages.length > 1) {
    throw new Error(
      "baicai v1 supports at most 1 reference image. Remove extras or choose another provider.",
    );
  }
}

export function buildRequestBody(
  prompt: string,
  model: string,
  args: CliArgs,
  refUrl: string | null,
): BaicaiRequestBody {
  const taskType = detectTaskType(model, args);
  const family = getModelFamily(model);

  if (family === "open-source") {
    const { width, height } = resolveOpenSourceSize(args);
    const input: OpenSourceInput = {
      prompt,
      number_of_images: 1,
      width,
      height,
    };
    if (taskType !== "txt2img") {
      if (!refUrl) {
        throw new Error("baicai img2img/image-edit requires a reference image URL.");
      }
      input.image = refUrl;
    }
    return {
      selected_model: model,
      task_type: taskType,
      response_format: "url",
      input,
    };
  }

  const input: CommercialInput = {
    prompt,
    number_of_images: 1,
    resolution: resolveCommercialResolution(args),
    aspect_ratio: args.aspectRatio ?? "1:1",
  };
  if (taskType !== "txt2img") {
    if (!refUrl) {
      throw new Error("baicai img2img/image-edit requires a reference image URL.");
    }
    input.image = refUrl;
  }
  return {
    selected_model: model,
    task_type: taskType,
    response_format: "url",
    input,
  };
}

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

export async function uploadReferenceImage(
  localPathOrUrl: string,
  apiKey: string,
  baseUrl: string,
): Promise<string> {
  if (isHttpUrl(localPathOrUrl)) return localPathOrUrl;

  const resolved = path.resolve(localPathOrUrl);
  if (!existsSync(resolved)) {
    throw new Error(`baicai reference image not found: ${resolved}`);
  }

  const bytes = await readFile(resolved);
  const filename = path.basename(resolved);
  const blob = new Blob([new Uint8Array(bytes)]);
  const form = new FormData();
  form.append("file", blob, filename);

  const response = await fetch(`${baseUrl}/v1/resources/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`baicai upload error (${response.status}): ${err}`);
  }

  const body = (await response.json()) as BaicaiUploadResponse;
  if (body.code !== 0 || !body.data?.fullPath) {
    throw new Error(`baicai upload failed: ${body.message ?? "missing fullPath"}`);
  }
  return body.data.fullPath;
}

export async function pollTaskUntilComplete(
  taskId: string,
  apiKey: string,
  baseUrl: string,
  deps: PollDeps = defaultPollDeps,
): Promise<void> {
  for (let attempt = 0; attempt < POLL_MAX_ATTEMPTS; attempt += 1) {
    const response = await fetch(`${baseUrl}/v1/comfyui/tasks/${taskId}/status`, {
      method: "GET",
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`baicai status poll error (${response.status}): ${err}`);
    }
    const body = (await response.json()) as BaicaiStatusResponse;
    const status = body.data?.[0]?.status;
    if (status === "COMPLETED") return;
    if (status === "FAILED" || status === "CANCELED") {
      throw new Error(`baicai task ${taskId} ended with status ${status}`);
    }
    await deps.sleep(POLL_INTERVAL_MS);
  }
  throw new Error(
    `baicai task ${taskId} polling timed out after ${(POLL_INTERVAL_MS * POLL_MAX_ATTEMPTS) / 60_000} minutes.`,
  );
}

export async function fetchTaskResult(
  taskId: string,
  apiKey: string,
  baseUrl: string,
): Promise<string> {
  const response = await fetch(`${baseUrl}/v1/comfyui/tasks/${taskId}/result`, {
    method: "GET",
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`baicai task result error (${response.status}): ${err}`);
  }
  const body = (await response.json()) as BaicaiResultResponse;
  const url = extractUrlFromResultBody(body);
  if (!url) {
    throw new Error(`baicai task ${taskId} result missing image URL`);
  }
  return url;
}

export function extractUrlFromResultBody(body: BaicaiResultResponse): string | null {
  if (!body.data) return null;
  if (Array.isArray(body.data)) {
    const entry = body.data[0];
    if (!entry) return null;
    return entry.result?.urls?.[0] ?? entry.urls?.[0] ?? null;
  }
  return body.data.urls?.[0] ?? body.data.result?.urls?.[0] ?? null;
}

export function extractImmediateUrl(body: BaicaiImmediateResponse | BaicaiTaskResponse): string | null {
  const data = (body as BaicaiImmediateResponse).data;
  if (!data || Array.isArray(data)) return null;
  return data.url?.[0] ?? null;
}

export function extractTaskId(body: BaicaiTaskResponse): string | null {
  const data = body.data;
  if (!Array.isArray(data) || data.length === 0) return null;
  return data[0]?.taskId ?? null;
}

async function downloadImage(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download image from baicai: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function generateImage(
  prompt: string,
  model: string,
  args: CliArgs,
  deps: PollDeps = defaultPollDeps,
): Promise<Uint8Array> {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error("BAICAI_API_KEY is required. Get one from https://cloud.baicaiinfer.com/.");
  }
  const baseUrl = getBaseUrl();

  validateArgs(model, args);

  let refUrl: string | null = null;
  const localRef = args.referenceImages[0];
  if (localRef) {
    refUrl = await uploadReferenceImage(localRef, apiKey, baseUrl);
  }

  const body = buildRequestBody(prompt, model, args, refUrl);

  const response = await fetch(`${baseUrl}/v1/images/generations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`baicai API error (${response.status}): ${err}`);
  }

  const parsed = (await response.json()) as BaicaiImmediateResponse & BaicaiTaskResponse;
  if (parsed.code != null && parsed.code !== 0) {
    throw new Error(`baicai API error: ${parsed.message ?? `code ${parsed.code}`}`);
  }

  const immediateUrl = extractImmediateUrl(parsed);
  if (immediateUrl) {
    return downloadImage(immediateUrl);
  }

  const taskId = extractTaskId(parsed);
  if (!taskId) {
    throw new Error("Unexpected baicai response shape: missing url and taskId.");
  }

  await pollTaskUntilComplete(taskId, apiKey, baseUrl, deps);
  const resultUrl = await fetchTaskResult(taskId, apiKey, baseUrl);
  return downloadImage(resultUrl);
}
