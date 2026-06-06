import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.rag_retriever.rag_module import (
    create_sample_corpus,
    build_faiss_index,
    retrieve_reference,
    retrieve_with_metadata
)

def run_rag_tests():
    print("=== BẮT ĐẦU KIỂM THỬ RAG MODULE ===")
    
    # 1. Tạo file dữ liệu mẫu
    print("\n[1] Đang tạo Corpus câu trả lời mẫu (Reference Answers)...")
    corpus_path = "data/reference_answers.jsonl"
    create_sample_corpus(corpus_path)
    print(f"  -> Đã tạo xong file tại: {corpus_path}")

    # 2. Xây dựng Vector Index bằng FAISS
    # Lưu ý: Lần đầu chạy sẽ tốn khoảng 10-20 giây để tải model 'all-MiniLM-L6-v2' từ HuggingFace
    print("\n[2] Đang xây dựng Vector Index bằng FAISS (Có thể mất chút thời gian tải model)...")
    build_faiss_index(corpus_path)
    print("  -> Đã xây dựng và lưu index thành công.")

    # 3. Test Truy vấn 1: CV chuẩn DevOps
    print("\n[3] TEST TRUY VẤN 1: Ứng viên DevOps / Backend")
    cv_devops = "I have strong experience deploying applications using Docker containers and orchestrating them with Kubernetes in a CI/CD pipeline."
    jd_devops = "Looking for a Senior Cloud Engineer to manage containerized workloads."
    
    # Lấy top 1 answer
    best_answer = retrieve_reference(cv_text=cv_devops, job_description=jd_devops)
    print(f"  > CV Input: {cv_devops[:60]}...")
    print(f"  > Kết quả Reference lý tưởng nhất:\n    '{best_answer}'")

    # 4. Test Truy vấn 2: CV Data Science (Dùng retrieve_with_metadata để xem chi tiết khoảng cách Vector)
    print("\n[4] TEST TRUY VẤN 2: Ứng viên Data Science / AI (Xem chi tiết Top 3)")
    cv_ai = "Skilled in Python programming and training Deep Learning models using gradient descent optimization."
    jd_ai = "Data Scientist required to build predictive models."
    
    top_results = retrieve_with_metadata(cv_text=cv_ai, job_description=jd_ai, top_k=3)
    print(f"  > CV Input: {cv_ai[:60]}...")
    for i, (text, meta, dist) in enumerate(top_results, 1):
        # dist (L2 distance) càng nhỏ (gần 0) thì càng giống nhau
        print(f"  --- Top {i} (Khoảng cách L2: {dist:.4f}) ---")
        print(f"  Topic: {meta.get('topic')}")
        print(f"  Answer: {text}")

if __name__ == "__main__":
    # Tắt cảnh báo của thư viện transformers để output sạch hơn
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        run_rag_tests()
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG]: {e}")
        print("Gợi ý: Hãy chắc chắn bạn đã chạy 'pip install faiss-cpu sentence-transformers'")