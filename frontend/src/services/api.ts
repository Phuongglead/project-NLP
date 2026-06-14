import axios, { AxiosInstance } from "axios";
import { GeneratedQuestion } from "../types";
import { getApiBaseUrl, loadAppConfig } from "../config/apiConfig";

export type Specialization =
  | "backend"
  | "frontend"
  | "devops"
  | "ml"
  | "mobile"
  | "data_engineer"
  | "qa"
  | "other";

export type ExperienceLevel = "junior" | "middle" | "senior";
export type PromptMode = "cv" | "specialization" | "mixed";

const MIN_LOADING_MS = 450;

let clientPromise: Promise<AxiosInstance> | null = null;

async function getClient(): Promise<AxiosInstance> {
  if (!clientPromise) {
    clientPromise = getApiBaseUrl().then((baseURL) =>
      axios.create({ baseURL, timeout: 300_000 })
    );
  }
  return clientPromise;
}

/** Call after changing API host/port so the next request uses the new base URL. */
export function resetApiClient(): void {
  clientPromise = null;
}

/** Keep spinner visible briefly while server-side fallbacks run (no UI message). */
export async function withMinLoading<T>(fn: () => Promise<T>): Promise<T> {
  const started = Date.now();
  try {
    return await fn();
  } finally {
    const elapsed = Date.now() - started;
    if (elapsed < MIN_LOADING_MS) {
      await new Promise((r) => setTimeout(r, MIN_LOADING_MS - elapsed));
    }
  }
}

export interface UploadCVResponse {
  cv_session_id: string;
  extracted_text_preview: string;
  review_mode?: boolean;
  holdout_cv_id?: string | null;
  figure_png_url?: string | null;
  figure_pdf_url?: string | null;
}

export const uploadCV = async (file: File): Promise<UploadCVResponse> => {
  await loadAppConfig();
  const client = await getClient();
  const form = new FormData();
  form.append("file", file);
  const res = await client.post<UploadCVResponse>("/interview/upload-cv", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export interface GenerateRequest {
  specialization: Specialization;
  experience_level: ExperienceLevel;
  mode: PromptMode;
  custom?: {
    tech_stack?: string;
    company_profile?: string;
    job_description?: string;
  };
  cv_session_id?: string;
  num_questions: number;
  generate_with_llm?: boolean;
}

export interface GenerateResponse {
  questions: GeneratedQuestion[];
  used_mode: PromptMode;
  cv_summary?: string | null;
  review_mode?: boolean;
  holdout_cv_id?: string | null;
  extracted_skills?: string[];
  figure_png_url?: string | null;
  figure_pdf_url?: string | null;
}

export const generateQuestions = async (body: GenerateRequest): Promise<GenerateResponse> => {
  await loadAppConfig();
  const client = await getClient();
  const res = await client.post<GenerateResponse>("/interview/generate", body);
  return res.data;
};

export interface FeedbackRequest {
  corpus_id: string;
  generated_question: string;
  rating: number;
  cv_session_id?: string;
  ideal_answer?: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  status: string;
}

export const submitFeedback = async (body: FeedbackRequest): Promise<FeedbackResponse> => {
  const client = await getClient();
  const res = await client.post<FeedbackResponse>("/interview/feedback", body);
  return res.data;
};
