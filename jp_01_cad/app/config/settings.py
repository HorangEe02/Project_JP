"""
DrawingLLM 설정 관리 모듈
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """애플리케이션 전역 설정"""

    # === 프로젝트 경로 ===
    project_root: Path = Path(__file__).parent.parent
    upload_dir: Path = Field(default=Path("./data/sample_drawings"))
    chroma_persist_dir: Path = Field(default=Path("./data/vector_store"))

    # === Ollama ===
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"

    # === 임베딩 모델 ===
    clip_model: str = "ViT-B/32"
    clip_finetuned_path: str = "./models/clip_finetuned.pt"  # Fine-tuned CLIP 체크포인트
    text_embedding_model: str = "intfloat/multilingual-e5-small"

    # === 벡터 DB ===
    chroma_collection_name: str = "drawings"

    # === 검색 ===
    search_top_k: int = 10
    image_weight: float = 0.15  # Fine-tuned CLIP으로 이미지 검색 활성화
    text_weight: float = 0.85

    # === YOLOv8-cls 분류기 ===
    yolo_cls_model_path: str = "./models/yolo_cls_v2_best.pt"
    yolo_cls_confidence_threshold: float = 0.5
    yolo_cls_enabled: bool = True
    yolo_cls_device: str = ""  # 빈 문자열이면 자동 선택

    # === YOLOv8-det 객체탐지기 ===
    yolo_det_model_path: str = "./models/yolo_det_best.pt"
    yolo_det_confidence_threshold: float = 0.3  # det는 recall 우선 → cls(0.5)보다 낮게
    yolo_det_enabled: bool = True
    yolo_det_device: str = ""  # 빈 문자열이면 자동 선택
    yolo_det_iou_threshold: float = 0.5  # NMS IoU 임계값

    # === Phase 4: LLM 컨텍스트 주입 ===
    llm_context_injection: bool = True       # YOLO/OCR 컨텍스트를 LLM에 주입
    llm_text_only_mode: bool = True          # 충분한 컨텍스트 시 이미지 없이 분석
    llm_hallucination_check: bool = True     # 환각 검증 활성화
    llm_num_predict_describe: int = 4096     # describe 토큰 (기존 8192 → 감소)
    llm_num_predict_metadata: int = 1024     # metadata 토큰
    llm_num_predict_qa: int = 2048           # Q&A 토큰

    # === 보안: 모델 무결성 검증 ===
    yolo_cls_sha256: str = ""   # YOLOv8-cls 모델 SHA256 (빈 문자열이면 스킵)
    yolo_det_sha256: str = ""   # YOLOv8-det 모델 SHA256 (빈 문자열이면 스킵)

    # === 보안: LLM 레이트 리미팅 ===
    llm_rate_limit_rpm: int = 30   # 분당 최대 LLM 호출 횟수 (0이면 무제한)

    # === 보안: 로그 로테이션 ===
    log_rotation: str = "50 MB"    # 로그 파일 회전 크기
    log_retention: str = "7 days"  # 로그 보관 기간
    log_file: str = "logs/drawingllm.log"  # 로그 파일 경로

    # === 카테고리 키워드 (검색 임베딩 보강) ===
    category_keywords_path: str = "./data/category_keywords.json"

    # === 파일 경로 매핑 (Docker용) ===
    # 원본 도면 경로의 접두사(호스트 경로)를 컨테이너 경로로 치환
    # 예: /Volumes/ExtDrive/data → /app/data/sample_drawings
    drawing_path_remap_from: str = "/Volumes/Corsair EX300U Media/00_work_out/02_ing/CAD/data/"
    drawing_path_remap_to: str = "/Volumes/Corsair EX300U Media/00_work_out/01_complete/me/01_CAD/data/"

    # === 파일 처리 ===
    max_file_size_mb: int = 50
    supported_formats: list[str] = ["png", "jpg", "jpeg", "pdf", "tiff", "tif"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
