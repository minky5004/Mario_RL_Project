# Mario RL — Python 게임 환경 서버 (헤드리스)
#
# 구버전 gym-super-mario-bros 스택은 Python 3.10 에서 안정적이므로 해당 이미지를 사용한다.
# nes-py 는 C++ 확장을 컴파일하므로 build-essential 이 필요하다.
#
# 빌드:  docker build -t mario-env ./python
# 실행:  docker run --rm -p 9999:9999 mario-env
#   (그 다음 Windows 에서  ./gradlew run  으로 Java 클라이언트 실행)

FROM python:3.10-slim

# nes-py(C++ 확장) 컴파일에 필요한 빌드 도구
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 코드 복사
COPY mario_env_server.py .

# 헤드리스 기본 (화면 렌더링 off). 필요하면 docker run -e RENDER=1 ...
ENV RENDER=0

EXPOSE 9999
CMD ["python", "mario_env_server.py"]
