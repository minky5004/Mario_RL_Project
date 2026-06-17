"""Mario RL — Python 게임 환경 서버 (헤드리스/Docker용).

역할
----
- ``gym-super-mario-bros`` 로 슈퍼마리오 1-1 을 실행한다.
- TCP 소켓(포트 9999)으로 Java 강화학습 클라이언트와 통신한다.
- 매 스텝 게임 상태(GameState)를 JSON 으로 보내고, Java 가 보낸 행동(Action)을 게임에 적용한다.

통신 프로토콜 (Java 와의 계약 — 순서를 어기면 데드락)
------------------------------------------------------
1. 서버(Python)가 ``accept`` 후 **첫 상태를 먼저 전송**한다 (reset 직후).
2. 이후 반복:  Java 행동 수신 → ``env.step`` → 다음 상태 전송.
3. ``done`` 상태를 보낸 뒤에는 **행동을 받지 않고** 곧장 ``reset`` 하여
   다음 에피소드의 첫 상태를 전송한다.
4. 메시지는 ``\\n`` 으로 구분, 인코딩은 UTF-8.

실행 환경
---------
- Docker(Python 3.10) 컨테이너에서 **헤드리스**로 실행하는 것을 기본으로 한다.
  화면 렌더링은 기본 비활성(환경변수 ``RENDER=1`` 로 켤 수 있음 — WSL/로컬용).
- ``HOST="0.0.0.0"`` 으로 바인딩해야 ``docker run -p 9999:9999`` 로 호스트(Windows)의
  Java 가 ``localhost:9999`` 로 접속할 수 있다.
"""

import io
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from PIL import Image

# --- 통신 설정 ---------------------------------------------------------------
HOST = "0.0.0.0"   # 컨테이너 외부(호스트)에서 접속 가능하도록 모든 인터페이스에 바인딩
PORT = 9999

# 화면 렌더링 여부 (헤드리스 기본 OFF). WSL/로컬에서 보고 싶으면 RENDER=1.
RENDER = os.environ.get("RENDER", "0") == "1"

# --- 브라우저 화면 스트리밍 설정 ---------------------------------------------
# 게임 화면(obs RGB)을 MJPEG 로 송출한다. 브라우저에서 http://localhost:8081 로 본다.
# (env.render() 와 무관 — obs 픽셀을 직접 인코딩하므로 헤드리스에서도 동작)
STREAM_PORT = 8081
STREAM_SCALE = 3        # 240x256 화면을 3배 확대해 보기 좋게
STREAM_FPS = 30

# --- 행동 매핑 ---------------------------------------------------------------
# Java 의 Action enum(0~5)과 1:1 로 맞춘 nes-py 버튼 조합.
#   0 NOOP, 1 RIGHT, 2 RIGHT_JUMP, 3 JUMP, 4 LEFT, 5 RIGHT_RUN_JUMP
#   (A = 점프, B = 달리기)
CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
    ["left"],
    ["right", "B", "A"],
]

# --- 보상 상수 (docs/03-보상설계.md 의 표와 일치) ----------------------------
FORWARD_SCALE = 0.1      # 전진 1px 당 보상
BACKWARD_PENALTY = -5.0  # 후퇴 시
DEATH_PENALTY = -100.0   # 사망 시
CLEAR_REWARD = 1000.0    # 깃발 도달(클리어)
TIMEOUT_PENALTY = -50.0  # 시간 초과

# --- enemy_near 탐지 설정 ----------------------------------------------------
# 마리오는 스크롤 중 화면상 일정 x(~110px) 근처에 머문다. 그 오른쪽 영역에서
# 굼바 갈색 계열 픽셀이 보이면 "적이 가깝다"고 (대략) 판단한다. (휴리스틱)
ENEMY_SCAN_X = 110
ENEMY_SCAN_WIDTH = 40
ENEMY_PIXEL_THRESHOLD = 20


def step_compat(env, action):
    """gym 구/신 API 모두 지원하는 step 래퍼.

    구 API: ``obs, reward, done, info``  /  신 API: ``obs, reward, terminated, truncated, info``
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    return obs, reward, bool(done), info


def detect_enemy_near(obs):
    """관측 RGB 배열에서 마리오 앞쪽에 적(갈색 계열)이 있는지 대략 판단한다."""
    if obs is None:
        return False
    height, width, _ = obs.shape
    x0 = min(ENEMY_SCAN_X, width - 1)
    x1 = min(x0 + ENEMY_SCAN_WIDTH, width)
    region = obs[:, x0:x1, :]

    r = region[..., 0].astype(int)
    g = region[..., 1].astype(int)
    b = region[..., 2].astype(int)
    # 굼바 갈색 대략: R 140~220, G 50~130, B<100, 그리고 R>G>B
    mask = (r > 140) & (r < 220) & (g > 50) & (g < 130) & (b < 100) & (r > g) & (g > b)
    return bool(mask.sum() > ENEMY_PIXEL_THRESHOLD)


def compute_reward(prev_x, info, done):
    """docs/03 의 보상표대로 보상과 종료 원인 문자열을 계산한다."""
    x = int(info.get("x_pos", prev_x))
    dx = x - prev_x

    if dx > 0:
        reward = dx * FORWARD_SCALE
    elif dx < 0:
        reward = BACKWARD_PENALTY
    else:
        reward = 0.0

    status = "running"
    if info.get("flag_get"):              # 깃발 도달 = 클리어
        reward = CLEAR_REWARD
        status = "clear"
    elif done:
        if int(info.get("time", 1)) <= 0:  # 시간 초과
            reward = TIMEOUT_PENALTY
            status = "timeout"
        else:                              # 그 외 종료 = 사망
            reward = DEATH_PENALTY
            status = "dead"
    return reward, status


def build_state(info, reward, done, status, obs):
    """Java GameState 모델과 같은 필드의 dict 를 만든다 (snake_case)."""
    return {
        "mario_x": int(info.get("x_pos", 0)),
        "mario_y": int(info.get("y_pos", 0)),
        "enemy_near": detect_enemy_near(obs),
        "score": int(info.get("score", 0)),
        "time_left": int(info.get("time", 0)),
        "reward": float(reward),
        "done": bool(done),
        "info": status,
    }


def render_safe(env):
    """RENDER=1 일 때만 best-effort 로 렌더링한다 (헤드리스에선 호출 안 됨)."""
    if not RENDER:
        return
    try:
        env.render()
    except Exception:
        pass


# --- 브라우저 화면 스트리밍 (MJPEG) -----------------------------------------
# 최신 게임 프레임(numpy RGB). 단일 변수 대입은 GIL 하에서 원자적이라 락 없이 공유한다.
_latest_obs = None


def set_frame(obs):
    """최신 게임 화면을 스트리밍 버퍼에 저장한다."""
    global _latest_obs
    _latest_obs = obs


def _encode_jpeg(obs):
    """numpy RGB 배열을 (확대하여) JPEG 바이트로 인코딩한다."""
    img = Image.fromarray(obs)
    if STREAM_SCALE != 1:
        img = img.resize((img.width * STREAM_SCALE, img.height * STREAM_SCALE), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


_PAGE = (
    "<html><head><title>Mario RL</title></head>"
    "<body style='margin:0;background:#111;text-align:center'>"
    "<img src='/stream' style='image-rendering:pixelated;margin-top:10px'>"
    "</body></html>"
).encode("utf-8")


class _StreamHandler(BaseHTTPRequestHandler):
    """'/' 는 보기 페이지, '/stream' 은 MJPEG 스트림을 제공한다."""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)
            return
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        delay = 1.0 / STREAM_FPS
        try:
            while True:
                obs = _latest_obs
                if obs is not None:
                    jpg = _encode_jpeg(obs)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 브라우저 탭을 닫으면 발생 — 무시

    def log_message(self, *args):
        pass  # HTTP 접속 로그 억제


def start_stream_server():
    """백그라운드 스레드에서 MJPEG 스트리밍 서버를 띄운다."""
    server = ThreadingHTTPServer(("0.0.0.0", STREAM_PORT), _StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Stream] 브라우저에서 http://localhost:{STREAM_PORT} 로 화면 보기", flush=True)


def serve(conn, env):
    """연결된 Java 클라이언트와 에피소드를 무한 반복하며 통신한다."""
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")

    def send_state(state):
        wfile.write(json.dumps(state) + "\n")
        wfile.flush()

    while True:  # 에피소드 무한 반복 (항상 1-1)
        reset_result = env.reset()
        _obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        # reset 직후에는 info(x_pos 등)가 없으므로 NOOP 한 스텝으로 초기 정보를 얻는다.
        obs, _, done, info = step_compat(env, 0)
        render_safe(env)
        set_frame(obs)
        prev_x = int(info.get("x_pos", 0))

        # ① 첫 상태 전송 (아직 Java 행동 전 — 보상 0, done False)
        send_state(build_state(info, 0.0, False, "running", obs))

        while not done:
            # ② Java 행동 수신
            line = rfile.readline()
            if not line:
                raise ConnectionError("Java 클라이언트 연결이 종료되었습니다.")
            action = int(json.loads(line)["action"])

            # ③ 게임 한 스텝 진행
            obs, _, done, info = step_compat(env, action)
            render_safe(env)
            set_frame(obs)

            reward, status = compute_reward(prev_x, info, done)
            prev_x = int(info.get("x_pos", prev_x))

            # ④ 결과 상태 전송 (done 이면 종료 신호)
            send_state(build_state(info, reward, done, status, obs))
        # done 상태를 보냈으므로 recv 없이 위로 돌아가 reset → 첫 상태 전송


def main():
    env = None
    server = None
    try:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, CUSTOM_MOVEMENT)

        start_stream_server()  # 브라우저 화면 스트리밍 시작

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[Server] Java 연결 대기 중... {HOST}:{PORT}  (RENDER={RENDER})", flush=True)

        conn, addr = server.accept()
        print(f"[Server] Java 연결됨: {addr}", flush=True)
        with conn:
            serve(conn, env)
    except (ConnectionError, BrokenPipeError) as e:
        print(f"[Server] 연결 종료: {e}", flush=True)
    except KeyboardInterrupt:
        print("[Server] 사용자 종료(Ctrl+C)", flush=True)
    finally:
        if server is not None:
            server.close()
        if env is not None:
            env.close()
        print("[Server] 정리 완료, 종료합니다.", flush=True)


if __name__ == "__main__":
    main()
