import React, { useState } from "react";
import { uploadCV, generateQuestions, withMinLoading, Specialization, ExperienceLevel, PromptMode } from "./services/api";
import { GeneratedQuestion } from "./types";
import ApiSettingsPanel from "./components/ApiSettingsPanel";
import QuestionList from "./components/QuestionList";
import ReviewEvaluationTable from "./components/ReviewEvaluationTable";
import { exportQuestionsToPdf } from "./services/pdf";
import { submitFeedback } from "./services/api";

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [cvSessionId, setCvSessionId] = useState<string | null>(null);
  const [specialization, setSpecialization] = useState<Specialization>("backend");
  const [level, setLevel] = useState<ExperienceLevel>("middle");
  const [mode, setMode] = useState<PromptMode>("mixed");
  const [techStack, setTechStack] = useState("");
  const [companyProfile, setCompanyProfile] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [loading, setLoading] = useState(false);
  const [questions, setQuestions] = useState<GeneratedQuestion[]>([]);
  const [cvPreview, setCvPreview] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [reviewMode, setReviewMode] = useState(false);
  const [extractedSkills, setExtractedSkills] = useState<string[]>([]);
  const [holdoutCvId, setHoldoutCvId] = useState<string | null>(null);
  const [figurePngUrl, setFigurePngUrl] = useState<string | null>(null);
  const [figurePdfUrl, setFigurePdfUrl] = useState<string | null>(null);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState<Record<number, boolean>>({});

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CV file first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await withMinLoading(() => uploadCV(file));
      setCvSessionId(res.cv_session_id);
      setCvPreview(res.extracted_text_preview);
      setReviewMode(!!res.review_mode);
      setHoldoutCvId(res.holdout_cv_id ?? null);
      setFigurePngUrl(res.figure_png_url ?? null);
      setFigurePdfUrl(res.figure_pdf_url ?? null);
      setRatings({});
      setSubmitted({});
    } catch (e: any) {
      const msg =
        e?.message === "Network Error"
          ? "Cannot reach API (check host/port in API settings, or CORS if UI is on another machine)."
          : e?.message || "Failed to upload CV.";
      setError(msg);
      console.error("[upload-cv]", e);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    const effectiveMode = (!cvSessionId && (mode === "cv" || mode === "mixed")) ? "specialization" : mode;

    setError(null);
    setLoading(true);
    try {
      const res = await withMinLoading(() =>
        generateQuestions({
          cv_session_id: cvSessionId || undefined,
          specialization,
          experience_level: level,
          mode: effectiveMode as PromptMode,
          num_questions: numQuestions,
          custom: {
            tech_stack: techStack || undefined,
            company_profile: companyProfile || undefined,
            job_description: jobDescription || undefined
          }
        })
      );
      setQuestions(res.questions);
      setReviewMode(!!res.review_mode);
      setExtractedSkills(res.extracted_skills || []);
      setHoldoutCvId(res.holdout_cv_id ?? holdoutCvId);
      setFigurePngUrl(res.figure_png_url ?? figurePngUrl);
      setFigurePdfUrl(res.figure_pdf_url ?? figurePdfUrl);
      setRatings({});
      setSubmitted({});
    } catch (e: any) {
      const msg =
        e?.message === "Network Error"
          ? "Cannot reach API (check host/port in API settings, or CORS if UI is on another machine)."
          : e?.message || "Failed to generate questions.";
      setError(msg);
      console.error("[generate]", e);
    } finally {
      setLoading(false);
    }
  };

  const handleRate = async (idx: number, q: GeneratedQuestion, rating: number) => {
    setRatings((prev) => ({ ...prev, [idx]: rating }));
    if (!q.corpus_id) {
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
        cv_session_id: cvSessionId || undefined,
      });
      setSubmitted((prev) => ({ ...prev, [idx]: true }));
    } catch (err) {
      setSubmitted((prev) => ({ ...prev, [idx]: true }));
      console.warn(`[rating] API unavailable; stored locally for question ${idx + 1}.`, err);
    }
  };

  const handleExportPdf = () => {
    exportQuestionsToPdf(questions);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>SA-AQG Interview Question Generator</h1>
        <p>Skill-Aware Answer-Aware question generation: NER + FAISS RAG + Gemini.</p>
        <ApiSettingsPanel />
      </header>

      <main className={`app-main ${reviewMode ? "review-layout" : ""}`}>
        {reviewMode && (
          <div className="review-banner" role="status">
            Human review mode — upload three IT PDF resumes, generate five questions each, and rate every pair (1–5).
          </div>
        )}
        <section className="card">
          <h2>1. Upload CV (Optional)</h2>
          <p className="info">
            {reviewMode
              ? "Upload a PDF resume; a PNG preview and PDF copy are saved for the thesis table."
              : "You can skip this step and generate questions based on specialization and level only."}
          </p>
          <div className="field">
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <button onClick={handleUpload} disabled={loading || !file}>
              {loading ? "Uploading..." : "Upload"}
            </button>
          </div>
          {cvPreview && !reviewMode && (
            <div className="preview">
              <h3>Extracted CV Preview</h3>
              <pre>{cvPreview}</pre>
            </div>
          )}
          {cvSessionId && (
            <button
              className="secondary"
              onClick={() => {
                setCvSessionId(null);
                setCvPreview("");
                setFile(null);
                setHoldoutCvId(null);
                setFigurePngUrl(null);
                setFigurePdfUrl(null);
              }}
            >
              Clear CV
            </button>
          )}
        </section>

        <section className="card">
          <h2>2. Configure Interview</h2>
          <div className="grid-2">
            <div className="field">
              <label>Specialization</label>
              <select value={specialization} onChange={(e) => setSpecialization(e.target.value as Specialization)}>
                <option value="backend">Backend</option>
                <option value="frontend">Frontend</option>
                <option value="devops">DevOps</option>
                <option value="ml">ML</option>
                <option value="mobile">Mobile</option>
                <option value="data_engineer">Data Engineer</option>
                <option value="qa">QA</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="field">
              <label>Experience level</label>
              <select value={level} onChange={(e) => setLevel(e.target.value as ExperienceLevel)}>
                <option value="junior">Junior</option>
                <option value="middle">Middle</option>
                <option value="senior">Senior</option>
              </select>
            </div>
            <div className="field">
              <label>Prompt mode</label>
              <select value={mode} onChange={(e) => setMode(e.target.value as PromptMode)}>
                <option value="cv">CV-focused</option>
                <option value="specialization">Specialization-focused</option>
                <option value="mixed">Mixed</option>
              </select>
            </div>
            <div className="field">
              <label>Number of questions</label>
              <input
                type="number"
                min={1}
                max={10}
                value={numQuestions}
                onChange={(e) => setNumQuestions(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="grid-3">
            <div className="field">
              <label>Tech stack (optional)</label>
              <textarea value={techStack} onChange={(e) => setTechStack(e.target.value)} />
            </div>
            <div className="field">
              <label>Company profile (optional)</label>
              <textarea value={companyProfile} onChange={(e) => setCompanyProfile(e.target.value)} />
            </div>
            <div className="field">
              <label>Job description (optional)</label>
              <textarea value={jobDescription} onChange={(e) => setJobDescription(e.target.value)} />
            </div>
          </div>

          <button className="primary" onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate questions"}
          </button>
          {cvSessionId ? (
            <p className="info">CV uploaded — session ready. Generate will use your resume.</p>
          ) : (
            <p className="info">
              No CV uploaded. Questions will be generated based on specialization and level only.
            </p>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        <section className="card">
          <h2>3. Questions</h2>
          {questions.length > 0 ? (
            <>
              {!reviewMode && (
                <div className="actions">
                  <button onClick={handleExportPdf}>Export to PDF</button>
                </div>
              )}
              {reviewMode ? (
                <ReviewEvaluationTable
                  questions={questions}
                  extractedSkills={extractedSkills}
                  figurePngUrl={figurePngUrl}
                  figurePdfUrl={figurePdfUrl}
                  holdoutCvId={holdoutCvId}
                  ratings={ratings}
                  onRate={handleRate}
                  submitted={submitted}
                />
              ) : (
                <QuestionList questions={questions} cvSessionId={cvSessionId || undefined} />
              )}
            </>
          ) : (
            <p>No questions yet. Upload a CV and click &quot;Generate questions&quot;.</p>
          )}
        </section>
      </main>
    </div>
  );
};

export default App;
