import React, { useState } from "react";
import { GeneratedQuestion } from "../types";
import { submitFeedback } from "../services/api";

interface Props {
  questions: GeneratedQuestion[];
  cvSessionId?: string;
}

const StarRating: React.FC<{
  value: number;
  onChange: (rating: number) => void;
  disabled?: boolean;
}> = ({ value, onChange, disabled }) => (
  <div className="star-rating" role="group" aria-label="Rate this question">
    {[1, 2, 3, 4, 5].map((star) => (
      <button
        key={star}
        type="button"
        className={`star-btn ${star <= value ? "star-filled" : ""}`}
        onClick={() => onChange(star)}
        disabled={disabled}
        aria-label={`${star} star`}
      >
        ★
      </button>
    ))}
  </div>
);

const QuestionList: React.FC<Props> = ({ questions, cvSessionId }) => {
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState<Record<number, boolean>>({});
  const [localOnly, setLocalOnly] = useState<Record<number, boolean>>({});

  const handleRate = async (idx: number, q: GeneratedQuestion, rating: number) => {
    setRatings((prev) => ({ ...prev, [idx]: rating }));

    if (!q.corpus_id) {
      setLocalOnly((prev) => ({ ...prev, [idx]: true }));
      setSubmitted((prev) => ({ ...prev, [idx]: true }));
      console.info(`[rating] Saved locally (no corpus_id) for question ${idx + 1}: ${rating}`);
      return;
    }

    try {
      await submitFeedback({
        corpus_id: q.corpus_id,
        generated_question: q.question,
        ideal_answer: q.ideal_answer,
        rating,
        cv_session_id: cvSessionId,
      });
      setSubmitted((prev) => ({ ...prev, [idx]: true }));
      setLocalOnly((prev) => {
        const next = { ...prev };
        delete next[idx];
        return next;
      });
    } catch (err) {
      setLocalOnly((prev) => ({ ...prev, [idx]: true }));
      setSubmitted((prev) => ({ ...prev, [idx]: true }));
      console.warn(`[rating] API unavailable; stored locally for question ${idx + 1}.`, err);
    }
  };

  return (
    <div className="question-list">
      {questions.map((q, idx) => (
        <article key={idx} className="question-card">
          <header>
            <span className={`pill pill-${q.difficulty}`}>{q.difficulty}</span>
            {q.category && <span className="pill pill-category">{q.category}</span>}
            {q.question_source && (
              <span className="pill pill-source">{q.question_source}</span>
            )}
            {q.skill && <span className="pill pill-skill">{q.skill}</span>}
          </header>
          <h3>{q.question}</h3>

          <details className="answer-collapse">
            <summary>Show ideal answer &amp; explanation</summary>
            <div className="answer-collapse-body">
              <p className="label">Ideal answer</p>
              <p>{q.ideal_answer}</p>
              <p className="label">Explanation</p>
              <p>{q.explanation}</p>
            </div>
          </details>

          <div className="feedback-row">
            <p className="label">Rate Q&amp;A usefulness (1–5)</p>
            <StarRating
              value={ratings[idx] || 0}
              onChange={(r) => handleRate(idx, q, r)}
              disabled={submitted[idx]}
            />
            {submitted[idx] && !localOnly[idx] && (
              <span className="feedback-saved">Saved</span>
            )}
            {submitted[idx] && localOnly[idx] && (
              <span className="feedback-saved feedback-local">Saved locally</span>
            )}
          </div>
        </article>
      ))}
    </div>
  );
};

export default QuestionList;
