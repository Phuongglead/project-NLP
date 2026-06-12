import React from "react";
import { GeneratedQuestion } from "../types";

interface Props {
  questions: GeneratedQuestion[];
}

const QuestionList: React.FC<Props> = ({ questions }) => {
  return (
    <div className="question-list">
      {questions.map((q, idx) => (
        <article key={idx} className="question-card">
          <header>
            <span className={`pill pill-${q.difficulty}`}>{q.difficulty}</span>
            {q.category && <span className="pill pill-category">{q.category}</span>}
          </header>
          <h3>{q.question}</h3>
          <p className="label">Ideal answer</p>
          <p>{q.ideal_answer}</p>
          <p className="label">Explanation</p>
          <p>{q.explanation}</p>
        </article>
      ))}
    </div>
  );
};

export default QuestionList;





