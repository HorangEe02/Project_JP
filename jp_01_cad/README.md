# CAD Vision — AI Drawing Search Engine

산업용 CAD 도면을 AI로 분류·검색·분석하는 풀스택 시스템

## Overview

| 항목 | 내용 |
|------|------|
| **데이터** | 9개 소스, 72,857건 산업용 도면 이미지 |
| **분류** | YOLOv8-cls v2 — 81 카테고리, Top-1 93.87% |
| **검색** | CLIP Fine-tuned + E5 하이브리드 (image 0.15 / text 0.85) |
| **벡터DB** | ChromaDB 68,647건 등록, Recall@5 = 0.372 |
| **탐지** | YOLOv8-det mAP50 = 0.552 |
| **분석** | Ollama VLM (Qwen2.5-VL) |
| **프론트엔드** | Streamlit 풀스택 웹 애플리케이션 |
| **배포** | Docker Compose |

## Architecture

```
[도면 이미지] → YOLOv8-cls (81 카테고리 분류)
             → CLIP Fine-tuned (이미지 임베딩)
             → EasyOCR (텍스트 추출)
                    ↓
             ChromaDB 벡터 DB (68,647건)
                    ↓
             하이브리드 검색 (CLIP 0.15 + E5 0.85)
                    ↓
             Streamlit 대시보드 (검색/분석/등록)
```

## Project Structure

```
├── app/                    # 메인 애플리케이션
│   ├── app/                # Streamlit UI
│   ├── config/             # 설정 (settings.py)
│   ├── core/               # 핵심 엔진
│   │   ├── classifier.py   # YOLOv8 분류기
│   │   ├── detector.py     # YOLOv8 탐지기
│   │   ├── embeddings.py   # CLIP/E5 임베딩
│   │   ├── vector_store.py # ChromaDB 관리
│   │   ├── pipeline.py     # 통합 파이프라인
│   │   ├── llm.py          # VLM 분석 (Ollama)
│   │   └── ocr.py          # EasyOCR 텍스트 추출
│   ├── scripts/            # 유틸리티 스크립트
│   ├── docs/               # 가이드 문서
│   ├── Dockerfile          # Docker 빌드
│   └── docker-compose.yml  # Docker Compose 배포
├── training/               # 학습 파이프라인
│   ├── scripts/            # 학습 스크립트
│   ├── PROJECT_GUIDE.md    # 학습 가이드
│   └── preprocessed_dataset/ # 데이터셋 메타정보
└── data/                   # 카테고리 매핑 메타데이터
```

## Key Results

### YOLOv8 Classification v2
- 81 카테고리, Train 49,337 / Val 6,142 / Test 6,232
- **Test Top-1 Accuracy: 93.87%** | Top-5: 98.04%
- Best epoch 70/90, 학습시간 15.7시간

### CLIP Fine-tuning
- InfoNCE 대조학습, Image tower frozen
- Image→Text R@5: 0.7% → **11.6% (16배 향상)**
- Text→Image R@5: 0.9% → **12.8% (14배 향상)**

### Metadata Enrichment
- 부품번호 추출: 60,456건 (파일명 기반)
- 재질 매핑: 60,721건 (카테고리-재질 자동 매핑, 75 항목)

## Tech Stack

| 영역 | 기술 |
|------|------|
| Language | Python 3.11 |
| Deep Learning | PyTorch, YOLOv8, CLIP (ViT-B/32), E5-multilingual-small |
| Vector DB | ChromaDB |
| OCR | EasyOCR |
| VLM | Ollama + Qwen2.5-VL |
| Frontend | Streamlit |
| Deploy | Docker, Docker Compose |

## Quick Start

```bash
# Local
cd app
pip install -r requirements.txt
streamlit run app/streamlit_app.py --server.port 8502

# Docker
cd app
docker compose up -d --build
# → http://localhost:8501
```

## Documentation

- [`app/README.md`](app/README.md) — 메인 앱 README
- [`app/PROJECT_SPEC.md`](app/PROJECT_SPEC.md) — 프로젝트 기능 명세 (v3.2)
- [`app/docs/GUIDE_DEVELOPER.md`](app/docs/GUIDE_DEVELOPER.md) — 개발자 가이드
- [`app/docs/GUIDE_USER.md`](app/docs/GUIDE_USER.md) — 사용자 가이드
- [`training/PROJECT_GUIDE.md`](training/PROJECT_GUIDE.md) — 학습 파이프라인 가이드

## Note

> 이미지 데이터, 모델 가중치(`.pt`), 벡터DB는 저작권 및 용량 문제로 포함되지 않습니다.
> 모델과 데이터는 별도 준비가 필요합니다.

---

**v3.2** | 2026-03-12
