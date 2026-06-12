export type QuestionDifficulty = "easy" | "medium" | "hard";

export interface GeneratedQuestion {
  question: string;
  ideal_answer: string;
  explanation: string;
  difficulty: QuestionDifficulty;
  category?: string | null;
}





