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
    corpus_path = args.corpus or "data/reference_answers.jsonl"
    if not os.path.exists(corpus_path):
        logger.info("No corpus found — creating sample corpus...")
        create_sample_corpus(corpus_path)
    build_faiss_index(corpus_path)


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
    p_idx.add_argument("--corpus", default=None, help="Path to reference answers JSONL")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
