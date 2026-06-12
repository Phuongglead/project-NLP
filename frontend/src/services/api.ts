import axios from "axios";
import { GeneratedQuestion } from "../types";

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

const client = axios.create({
  baseURL: "/api"
});

export interface UploadCVResponse {
  cv_session_id: string;
  extracted_text_preview: string;
}

export const uploadCV = async (file: File): Promise<UploadCVResponse> => {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post<UploadCVResponse>("/interview/upload-cv", form, {
    headers: { "Content-Type": "multipart/form-data" }
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
  generate_with_llm?: boolean;  // Optional: generate additional questions via LLM
}

export interface GenerateResponse {
  questions: GeneratedQuestion[];
  used_mode: PromptMode;
  cv_summary?: string | null;
}

export const generateQuestions = async (body: GenerateRequest): Promise<GenerateResponse> => {
  const res = await client.post<GenerateResponse>("/interview/generate", body);
  return res.data;
};



