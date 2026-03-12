#!/bin/bash
# ============================================================
#  CAD Vision — Docker 시작 스크립트
#
#  Docker Desktop for Mac에서 공백 포함 경로의 bind mount 오류를
#  Named Volume으로 우회하여 해결합니다.
#
#  사용법:
#    chmod +x docker-start.sh
#    ./docker-start.sh            # 빌드 + 시작
#    ./docker-start.sh --build    # 강제 재빌드 (캐시 없이)
#    ./docker-start.sh --down     # 중지 + 제거
#    ./docker-start.sh --reset    # 데이터 볼륨 초기화 + 재시작
# ============================================================

set -e

# --- 색상 출력 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== CAD Vision Docker 시작 ===${NC}"
echo ""

# --- 중지 옵션 ---
if [ "$1" = "--down" ]; then
    echo -e "${YELLOW}[*] Docker Compose 중지...${NC}"
    docker compose down
    echo -e "${GREEN}[✓] 중지 완료${NC}"
    exit 0
fi

# --- 리셋 옵션 ---
if [ "$1" = "--reset" ]; then
    echo -e "${RED}[*] 데이터 볼륨 초기화...${NC}"
    docker compose down
    docker volume rm cad-vision-data 2>/dev/null || true
    echo -e "${GREEN}[✓] 볼륨 삭제 완료 — 재빌드로 초기화됩니다${NC}"
    echo ""
fi

# --- 1. Docker 이미지 빌드 ---
echo -e "${YELLOW}[1/3] Docker 이미지 빌드...${NC}"
if [ "$1" = "--build" ]; then
    docker compose build --no-cache
else
    docker compose build
fi
echo ""

# --- 2. 기존 컨테이너 정리 ---
echo -e "${YELLOW}[2/3] 기존 컨테이너 정리...${NC}"
docker compose down 2>/dev/null || true
echo ""

# --- 3. 컨테이너 시작 ---
echo -e "${YELLOW}[3/3] 컨테이너 시작...${NC}"
docker compose up -d
echo ""

# --- 상태 확인 ---
echo -e "${GREEN}=== 시작 완료 ===${NC}"
echo ""
echo -e "   🌐 웹 UI:       ${CYAN}http://localhost:8501${NC}"
echo -e "   🤖 Ollama:      ${CYAN}http://localhost:11434${NC}"
echo ""
echo "   📋 로그 확인:   docker compose logs -f app"
echo "   🛑 중지:        ./docker-start.sh --down"
echo "   🔄 데이터 리셋: ./docker-start.sh --reset"
echo ""

# --- 헬스체크 (5초 대기) ---
echo -e "${YELLOW}[*] 컨테이너 상태 확인...${NC}"
sleep 5
if docker ps --filter "name=cad-vision-app" --format '{{.Status}}' | grep -q "Up"; then
    echo -e "   ${GREEN}✓ cad-vision-app: 실행 중${NC}"
else
    echo -e "   ${RED}✗ cad-vision-app: 시작 실패${NC}"
    echo "     로그: docker compose logs app"
fi

if docker ps --filter "name=cad-vision-ollama" --format '{{.Status}}' | grep -q "Up"; then
    echo -e "   ${GREEN}✓ cad-vision-ollama: 실행 중${NC}"
else
    echo -e "   ${RED}✗ cad-vision-ollama: 시작 실패${NC}"
fi

echo ""
echo -e "${YELLOW}📌 참고:${NC}"
echo "   • Docker에서는 Named Volume을 사용합니다 (공백 경로 우회)"
echo "   • ChromaDB 데이터(68,647건)는 이미지에서 자동 로드됩니다"
echo "   • 외부 도면 이미지는 Docker에서 제한적으로 표시됩니다"
echo "   • 전체 이미지 포함 개발 환경: localhost:8502 (로컬 Streamlit)"
