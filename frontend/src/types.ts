export type QuestionDifficulty = "easy" | "medium" | "hard";

export interface GeneratedQuestion {
  question: string;
  ideal_answer: string;
  explanation: string;
  difficulty: QuestionDifficulty;
  category?: string | null;
  corpus_id?: string | null;
  match_score?: number | null;
  question_source?: string | null;
  skill?: string | null;
  topic?: string | null;
}
