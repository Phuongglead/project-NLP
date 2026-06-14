"""
main.py — SA-AQG CLI Entry Point

Usage:
  # Single sample (stub mode for testing)
  SA_AQG_USE_STUBS=true python -m src.main run --cv "I know Kubernetes..." --job "DevOps Engineer"

  # Batch evaluation (100 samples)
  python -m src.main batch --input data/eval_samples.jsonl --n 100

  # Create sample eval data
  python -m src.main create-data --n 10

  # Train NER model (Member A)
  python -m src.main train-ner

  # Build RAG index (Member D)
  python -m src.main build-index

  # Run NLI evaluation on generated questions (Member B)
  python -m src.main eval-nli --input outputs/generated_questions.jsonl

  # Run full XAI evaluation (Member C)
  python -m src.main eval-xai --input outputs/generated_questions.jsonl
"""

import argparse
import sys
import os

from src.shared.utils.io_utils import get_logger

logger = get_logger("main")


def cmd_run(args):
    """Run pipeline on a single sample."""
    from src.pipeline.runner import run_pipeline
    result = run_pipeline(
        cv_text=args.cv,
        job_description=args.job,
        run_shap=not args.no_shap,
    )
    print("\n" + "="*60)
    print("SA-AQG PIPELINE RESULT")
    print("="*60)
    print(f"ID:               {result.id}")
    print(f"Skills extracted: {[e.entity for e in result.skills]}")
    print(f"Reference answer: {result.reference_answer[:100]}...")
    print(f"\nGenerated question:\n  {result.generated_question}")
    if result.evaluation:
        e = result.evaluation
        print(f"\nEvaluation:")
        print(f"  NLI:               {e.nli_label} ({e.nli_score:.3f})")
        print(f"  Citation Precision: {e.citation_precision:.3f}")
        print(f"  Citation Recall:    {e.citation_recall:.3f}")
        print(f"  SHAP CV Ratio:      {e.shap_cv_ratio:.3f}")
    print("="*60)


def cmd_batch(args):
    """Run full batch evaluation."""
    from src.pipeline.batch_processor import run_full_batch_evaluation
    metrics = run_full_batch_evaluation(
        input_path=args.input,
        n_samples=args.n,
        run_pipeline=not args.no_pipeline,
        run_baseline=not args.no_baseline,
        run_shap=args.shap,
    )
    print("\nBatch Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def cmd_create_data(args):
    """Create sample evaluation data."""
    from src.pipeline.batch_processor import create_sample_eval_data
    create_sample_eval_data(output_path=args.output, n=args.n)
    print(f"Sample data created: {args.output}")


def cmd_train_ner(args):
    """Train NER model (Member A)."""
    from src.core.ner_extractor.ner_module import train_ner
    train_ner()


def cmd_build_index(args):
    """Build RAG FAISS index (Member D)."""
    from src.core.rag_retriever.rag_module import create_sample_corpus, build_faiss_index
    from src.shared.utils.io_utils import load_config
    cfg = load_config()["rag"]
    corpus_path = args.corpus or cfg.get("reference_corpus_path", "data/knowledge_corpus.active.jsonl")
    index_path = args.output or cfg.get("index_path", "models/faiss_index.active")
    if not os.path.exists(corpus_path):
        logger.info("No corpus found — creating sample corpus...")
        create_sample_corpus(corpus_path)
    build_faiss_index(corpus_path, index_path=index_path)
    print(f"Index built: {index_path}.faiss from {corpus_path}")


def cmd_eval_nli(args):
    """Run NLI hallucination evaluation (Member B)."""
    from src.core.question_generator.nli_evaluator import run_nli_evaluation
    summary = run_nli_evaluation(
        questions_path=args.input,
        n_samples=args.n,
    )
    print("\nNLI Evaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_eval_download(args):
    """Download Kaggle IT resumes and export eval JSONL."""
    import json
    from pathlib import Path

    from scripts.eval.download_kaggle_resumes import (
        ROOT,
        download_dataset,
        export_it_resumes,
        _find_resume_csv,
    )
    from src.shared.corpus.schema import load_jsonl
    from src.shared.utils.io_utils import load_config

    cfg = load_config().get("evaluation", {})
    dataset_slug = cfg.get("kaggle_dataset", "snehaanbhawal/resume-dataset")
    category = cfg.get("it_category", "Information-Technology")
    job_desc = cfg.get(
        "default_job_description",
        "Information-Technology software engineering role.",
    )
    output = args.output or cfg.get("batch_input", "data/eval/it_resumes.jsonl")
    dataset_dir = ROOT / "data" / "kaggle" / "resume-dataset"
    output_path = ROOT / output if not os.path.isabs(output) else Path(output)

    if not args.skip_download:
        csv_path = download_dataset(dataset_slug, dataset_dir)
    else:
        csv_path = _find_resume_csv(dataset_dir)
        if not csv_path:
            raise FileNotFoundError(f"No CSV in {dataset_dir}")

    n = export_it_resumes(csv_path, output_path, category, job_desc)
    holdout_path = cfg.get("holdout_ids_path", "data/eval/holdout_cv_ids.json")
    holdout_file = ROOT / holdout_path if not os.path.isabs(holdout_path) else Path(holdout_path)
    holdout_file.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(str(output_path))
    holdout_ids = [r["cv_id"] for r in records[-3:]] if len(records) >= 3 else []
    with open(holdout_file, "w", encoding="utf-8") as f:
        json.dump(
            {"holdout_cv_ids": holdout_ids, "note": "Use these 3 CVs for human REVIEW_MODE eval"},
            f,
            indent=2,
        )
    print(f"Exported {n} resumes to {output_path}")
    print(f"Hold-out CV IDs -> {holdout_file}")


def cmd_eval_batch(args):
    """Run ALCE + SHAP batch evaluation on IT resumes."""
    import json as _json

    from scripts.eval.run_batch_eval import run_batch_eval
    from src.shared.corpus.schema import load_jsonl
    from src.shared.utils.io_utils import load_config

    cfg = load_config().get("evaluation", {})
    input_path = args.input or cfg.get("batch_input", "data/eval/it_resumes.jsonl")
    output_path = args.output or cfg.get("batch_output", "outputs/eval/batch_records.jsonl")
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(__file__), input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    n = run_batch_eval(
        input_path,
        output_path,
        n=args.n,
        shap_nsamples=args.shap_nsamples,
        rag_top_k=args.rag_top_k,
        skip_existing=not args.no_skip_existing,
    )
    print(f"Processed {n} CVs -> {output_path}")

    if not args.no_aggregate:
        from scripts.eval.aggregate_batch_report import aggregate
        from src.shared.corpus.schema import load_jsonl
        import json as _json

        xai_cfg = load_config().get("xai", {})
        eval_cfg = load_config().get("evaluation", {})
        summary_path = eval_cfg.get("batch_summary", "outputs/eval/batch_summary.json")
        if not os.path.isabs(summary_path):
            summary_path = os.path.join(os.path.dirname(__file__), summary_path)
        records = load_jsonl(output_path)
        summary = aggregate(
            records,
            cv_threshold=xai_cfg.get("cv_contribution_threshold", 0.40),
        )
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            _json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")


def cmd_eval_aggregate_batch(args):
    """Aggregate batch eval records into summary JSON."""
    from scripts.eval.aggregate_batch_report import main as aggregate_main

    argv = ["aggregate_batch_report.py"]
    if args.input:
        argv += ["--input", args.input]
    if args.output:
        argv += ["--output", args.output]
    old_argv = sys.argv
    sys.argv = argv
    try:
        aggregate_main()
    finally:
        sys.argv = old_argv


def cmd_eval_aggregate_review(args):
    """Aggregate REVIEW_MODE human ratings into summary JSON."""
    from scripts.eval.aggregate_review_report import main as aggregate_main

    argv = ["aggregate_review_report.py"]
    if args.input:
        argv += ["--input", args.input]
    if args.output:
        argv += ["--output", args.output]
    old_argv = sys.argv
    sys.argv = argv
    try:
        aggregate_main()
    finally:
        sys.argv = old_argv


def cmd_eval_prepare_holdout(args):
    """Export hold-out CVs as txt + PNG/PDF figures."""
    from scripts.eval.prepare_holdout_review import prepare_holdout

    manifest = prepare_holdout()
    for m in manifest:
        print(f"  {m['cv_id']}: {m['txt_path']}")


def cmd_eval_run_holdout_review(args):
    """Upload hold-out CVs and generate questions via API."""
    from scripts.eval.run_holdout_review import run_holdout_review

    run_holdout_review(api_base=args.api, num_questions=args.num_questions)


def cmd_eval_generate_review_latex(args):
    """Generate LaTeX human-review table and prose subsection."""
    from scripts.eval.generate_review_latex import finalize_review_report, generate_review_latex

    if getattr(args, "inject", False):
        result = finalize_review_report(
            review_path=args.input,
            output_dir=args.output_dir,
            thesis_figures=args.thesis_figures,
            experiments_path=getattr(args, "experiments_tex", None),
        )
        print(f"Injected -> {result['experiments_path']}")
    else:
        result = generate_review_latex(
            review_path=args.input,
            output_dir=args.output_dir,
            thesis_figures=args.thesis_figures,
        )
    print(f"Prose  -> {result['prose_path']}")
    print(f"Table  -> {result['table_path']}")
    print(f"Summary -> {result['summary_path']} (n_ratings={result['n_ratings']})")


def cmd_eval_xai(args):
    """Run XAI batch evaluation (Member C)."""
    from src.shared.utils.io_utils import load_jsonl
    from src.shared.contracts.schemas import GeneratorOutput
    from src.core.xai_evaluator.xai_module import evaluate_batch

    records = load_jsonl(args.input)[:args.n]
    gen_outputs = [GeneratorOutput.from_dict(r) for r in records]
    summary = evaluate_batch(
        gen_outputs,
        run_shap=not args.no_shap,
        shap_sample_size=args.shap_n,
    )
    print("\nXAI Evaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="SA-AQG: Skill-Aware Answer-Aware Question Generation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run pipeline on a single CV + job description")
    p_run.add_argument("--cv", required=True, help="CV text string")
    p_run.add_argument("--job", default="Software Engineer", help="Job description")
    p_run.add_argument("--no-shap", action="store_true", help="Skip SHAP attribution")
    p_run.set_defaults(func=cmd_run)

    # batch
    p_batch = sub.add_parser("batch", help="Run full batch evaluation (100 samples)")
    p_batch.add_argument("--input", default="data/eval_samples.jsonl")
    p_batch.add_argument("--n", type=int, default=None, help="Number of samples")
    p_batch.add_argument("--no-pipeline", action="store_true", help="Skip SA-AQG generation")
    p_batch.add_argument("--no-baseline", action="store_true", help="Skip baseline generation")
    p_batch.add_argument("--shap", action="store_true", help="Enable SHAP (slow)")
    p_batch.set_defaults(func=cmd_batch)

    # create-data
    p_data = sub.add_parser("create-data", help="Create sample evaluation data")
    p_data.add_argument("--output", default="data/eval_samples.jsonl")
    p_data.add_argument("--n", type=int, default=10)
    p_data.set_defaults(func=cmd_create_data)

    # train-ner
    p_ner = sub.add_parser("train-ner", help="Fine-tune NER model on SkillSpan")
    p_ner.set_defaults(func=cmd_train_ner)

    # build-index
    p_idx = sub.add_parser("build-index", help="Build FAISS RAG index")
    p_idx.add_argument("--corpus", default=None, help="Path to knowledge corpus JSONL")
    p_idx.add_argument("--output", default=None, help="FAISS index base path (without .faiss)")
    p_idx.set_defaults(func=cmd_build_index)

    # eval-nli
    p_nli = sub.add_parser("eval-nli", help="Run NLI hallucination evaluation")
    p_nli.add_argument("--input", required=True, help="Generated questions JSONL")
    p_nli.add_argument("--n", type=int, default=200)
    p_nli.set_defaults(func=cmd_eval_nli)

    # eval-xai
    p_xai = sub.add_parser("eval-xai", help="Run XAI evaluation (SHAP + ALCE)")
    p_xai.add_argument("--input", required=True, help="Generated questions JSONL")
    p_xai.add_argument("--n", type=int, default=50)
    p_xai.add_argument("--no-shap", action="store_true")
    p_xai.add_argument("--shap-n", type=int, default=50)
    p_xai.set_defaults(func=cmd_eval_xai)

    # eval-download
    p_edl = sub.add_parser("eval-download", help="Download Kaggle IT resumes for batch eval")
    p_edl.add_argument("--output", default=None)
    p_edl.add_argument("--skip-download", action="store_true")
    p_edl.set_defaults(func=cmd_eval_download)

    # eval-batch
    p_eb = sub.add_parser("eval-batch", help="Run ALCE+SHAP batch eval on IT resumes")
    p_eb.add_argument("--input", default=None)
    p_eb.add_argument("--output", default=None)
    p_eb.add_argument("--n", type=int, default=None)
    p_eb.add_argument("--shap-nsamples", type=int, default=None)
    p_eb.add_argument("--rag-top-k", type=int, default=5)
    p_eb.add_argument("--no-skip-existing", action="store_true")
    p_eb.add_argument("--no-aggregate", action="store_true")
    p_eb.set_defaults(func=cmd_eval_batch)

    # eval-aggregate-batch
    p_eab = sub.add_parser("eval-aggregate-batch", help="Aggregate batch eval JSONL to summary")
    p_eab.add_argument("--input", default=None)
    p_eab.add_argument("--output", default=None)
    p_eab.set_defaults(func=cmd_eval_aggregate_batch)

    # eval-aggregate-review
    p_ear = sub.add_parser("eval-aggregate-review", help="Aggregate REVIEW_MODE ratings")
    p_ear.add_argument("--input", default=None)
    p_ear.add_argument("--output", default=None)
    p_ear.set_defaults(func=cmd_eval_aggregate_review)

    # eval-prepare-holdout
    p_eph = sub.add_parser("eval-prepare-holdout", help="Prepare hold-out CV files and figures")
    p_eph.set_defaults(func=cmd_eval_prepare_holdout)

    # eval-run-holdout-review
    p_erh = sub.add_parser("eval-run-holdout-review", help="Run hold-out review via API")
    p_erh.add_argument("--api", default="http://127.0.0.1:8000")
    p_erh.add_argument("--num-questions", type=int, default=5)
    p_erh.set_defaults(func=cmd_eval_run_holdout_review)

    # eval-generate-review-latex
    p_egl = sub.add_parser("eval-generate-review-latex", help="Generate LaTeX review table")
    p_egl.add_argument("--input", default=None)
    p_egl.add_argument("--output-dir", default=None)
    p_egl.add_argument("--thesis-figures", default="figures")
    p_egl.add_argument("--inject", action="store_true", help="Insert into experiments.tex")
    p_egl.add_argument("--experiments-tex", default=None)
    p_egl.set_defaults(func=cmd_eval_generate_review_latex)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
