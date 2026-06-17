# Mario RL — Python 게임 환경 서버 (헤드리스)
#
# 빌드 컨텍스트는 저장소 루트다. (실제 소스는 python/ 에 있으므로 COPY 경로에 python/ 를 붙인다.)
# 컨텍스트 최소화를 위해 .dockerignore 로 python/ 외 파일은 제외한다.
#
# 빌드:  docker build -t mario-env .
# 실행:  docker run --rm -p 9999:9999 mario-env
#   (그 다음 Windows 에서  ./gradlew run  으로 Java 클라이언트 실행)

FROM python:3.10-slim

# nes-py(C++ 확장) 컴파일에 필요한 빌드 도구
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 코드 복사
COPY python/mario_env_server.py .

# 헤드리스 기본 (화면 렌더링 off). 필요하면 docker run -e RENDER=1 ...
ENV RENDER=0

EXPOSE 9999
CMD ["python", "mario_env_server.py"]
