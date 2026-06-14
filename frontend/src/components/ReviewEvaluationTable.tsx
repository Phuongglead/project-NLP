import React, { useMemo } from "react";
import { GeneratedQuestion } from "../types";

interface Props {
  questions: GeneratedQuestion[];
  extractedSkills: string[];
  figurePngUrl?: string | null;
  figurePdfUrl?: string | null;
  holdoutCvId?: string | null;
  ratings: Record<number, number>;
  onRate: (idx: number, question: GeneratedQuestion, rating: number) => void;
  submitted: Record<number, boolean>;
}

const StarRating: React.FC<{
  value: number;
  onChange: (rating: number) => void;
  disabled?: boolean;
}> = ({ value, onChange, disabled }) => (
  <div className="star-rating compact" role="group" aria-label="Rate this question">
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

const ReviewEvaluationTable: React.FC<Props> = ({
  questions,
  extractedSkills,
  figurePngUrl,
  figurePdfUrl,
  holdoutCvId,
  ratings,
  onRate,
  submitted,
}) => {
  const keywords = useMemo(() => {
    if (extractedSkills.length > 0) {
      return extractedSkills.slice(0, 12).join(", ");
    }
    const fromQuestions = questions
      .map((q) => q.skill)
      .filter((s): s is string => Boolean(s));
    return [...new Set(fromQuestions)].join(", ") || "—";
  }, [extractedSkills, questions]);

  const ratedValues = Object.values(ratings).filter((r) => r > 0);
  const average =
    ratedValues.length > 0
      ? (ratedValues.reduce((a, b) => a + b, 0) / ratedValues.length).toFixed(1)
      : "—";

  const cvLabel = holdoutCvId ? `CV ${holdoutCvId}` : "Uploaded CV";
  const cvPreview = figurePngUrl ? (
    figurePdfUrl ? (
      <a href={figurePdfUrl} target="_blank" rel="noreferrer" title="Open PDF resume">
        <img src={figurePngUrl} alt={cvLabel} className="review-cv-thumb" />
      </a>
    ) : (
      <img src={figurePngUrl} alt={cvLabel} className="review-cv-thumb" />
    )
  ) : (
    <span className="review-cv-placeholder">{cvLabel}</span>
  );

  return (
    <div className="review-table-wrap">
      <table className="review-table">
        <thead>
          <tr>
            <th>CV</th>
            <th>Extracted keywords</th>
            <th>Generated question</th>
            <th>Rating (1–5)</th>
          </tr>
        </thead>
        <tbody>
          {questions.map((q, idx) => (
            <tr key={idx}>
              {idx === 0 && (
                <>
                  <td rowSpan={questions.length + 1} className="review-cv-cell">
                    {cvPreview}
                  </td>
                  <td rowSpan={questions.length + 1} className="review-keywords-cell">
                    {keywords}
                  </td>
                </>
              )}
              <td className="review-question-cell">{q.question}</td>
              <td className="review-rating-cell">
                <StarRating
                  value={ratings[idx] || 0}
                  onChange={(r) => onRate(idx, q, r)}
                  disabled={submitted[idx]}
                />
              </td>
            </tr>
          ))}
          <tr className="review-mean-row">
            <td colSpan={1} className="review-mean-label">
              <strong>Mean</strong>
            </td>
            <td>
              <strong>{average}</strong>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ReviewEvaluationTable;
