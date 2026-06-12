import jsPDF from "jspdf";
import { GeneratedQuestion } from "../types";

export const exportQuestionsToPdf = (questions: GeneratedQuestion[]) => {
  const doc = new jsPDF();
  const margin = 10;
  const lineHeight = 6;
  let y = margin;

  doc.setFontSize(14);
  doc.text("AI Interview Questions", margin, y);
  y += lineHeight * 2;

  doc.setFontSize(10);
  questions.forEach((q, index) => {
    const lines: string[] = [];
    lines.push(`${index + 1}. [${q.difficulty.toUpperCase()}] ${q.question}`);
    lines.push(`Ideal answer: ${q.ideal_answer}`);
    lines.push(`Explanation: ${q.explanation}`);
    if (q.category) {
      lines.push(`Category: ${q.category}`);
    }
    lines.push(""); // spacer

    lines.forEach((line) => {
      const split = doc.splitTextToSize(line, 190);
      split.forEach((l: string) => {
        if (y > 280) {
          doc.addPage();
          y = margin;
        }
        doc.text(l, margin, y);
        y += lineHeight;
      });
    });
  });

  doc.save("interview-questions.pdf");
};





