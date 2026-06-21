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

# 프레임 스킵: 행동 1개를 몇 프레임 동안 유지할지. 값이 클수록 소켓 왕복이 줄어 한 판이 빨라진다.
# (1 = 스킵 없음, 4 = 표준) — Atari/마리오 강화학습의 표준 기법.
# [DAY5 트라이1] 점프 궤적 제어 해상도를 높이려 4→2 로 축소(한 판 ~2배 느려짐).
FRAME_SKIP = 2

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
DEATH_PENALTY = -100.0   # 사망 시 (적과 충돌 등)
CLEAR_REWARD = 1000.0    # 깃발 도달(클리어)
TIMEOUT_PENALTY = -50.0  # 시간 초과

# --- [DAY2] 낙사/굼바 사망 구분 (측정 인프라) -------------------------------
# 트라이1(낙사 페널티 -300)은 종료됨 → 보상은 baseline(-100)으로 복구.
# 단, 사망 원인(dead_pit/dead_enemy) 구분과 종료 로그는 "어디서 죽는지" 측정에 유용하므로 유지한다.
#   → 결과: docs/실험일지/[DAY2] 진짜 1차 관문 - 굼바 넘기.md
PIT_DEATH_PENALTY = -100.0   # 낙사 — baseline 사망과 동일 (트라이1 종료 후 복구)
# 마리오가 화면 아래로 떨어지면(=낙사) y_pos 가 이 값 이상이 된다. (실측: 낙사≈253 / 지상≈79)
PIT_Y_THRESHOLD = 200

# --- 적(굼바) 거리 탐지 설정 — [DAY2 트라이3] RAM 직접 읽기 -------------------
# 트라이2에서 갈색 픽셀 스캔(_is_enemy_brown)이 굼바가 아니라 "마리오 자신"을 잡고 있었음이
# 드러났다(항상 dist=3). → 픽셀 추측을 버리고 nes-py의 RAM(env.ram, 2KB)에서 적 슬롯의
# 실제 x좌표를 직접 읽어 "마리오 x − 적 x"로 거리를 계산한다. (SMB 표준 RAM 맵)
#
# SMB 적 슬롯은 5개. 각 슬롯마다 아래 주소를 가진다(전역 x = page*256 + x_in_page):
ENEMY_SLOTS = 5
ADDR_ENEMY_DRAWN = 0x0F     # 0x0F~0x13: 슬롯 활성 플래그(0=빈 슬롯, 그 외=적 존재)
ADDR_ENEMY_X_PAGE = 0x6E    # 0x6E~0x72: 적 x 상위 바이트(페이지)
ADDR_ENEMY_X = 0x87         # 0x87~0x8B: 적 x 하위 바이트(페이지 내 위치)
# 마리오 전역 x = info["x_pos"] (= RAM 0x6D*256 + 0x86) 와 같은 좌표계라 그대로 빼면 된다.
#
# 마리오 앞쪽 거리(px)를 4단계로 매핑. 굼바 폭 ~16px 기준, 점프 타이밍을 배우게끔 구간을 나눈다.
#   0 없음(앞에 적 없음/너무 멂) / 1 멂 / 2 가까움 / 3 위험함(바로 앞)
ENEMY_DIST_DANGER = 24   # 단계3 위험함: 마리오 앞 0~24px
ENEMY_DIST_NEAR = 48     # 단계2 가까움: 24~48px
ENEMY_DIST_FAR = 96      # 단계1 멂:     48~96px (그 너머는 0 없음)

# --- [DAY3 트라이1] 굼바 밟기(stomp) 보너스 --------------------------------
# 인식(enemy_dist)은 진짜가 됐으나(DAY2 트라이3) 보상이 "굼바 넘기"를 직접 가리키지 않아
# 첫 굼바를 '확신'하며 넘지 못했다(후반에도 20% 사망). → 밟기 자체에 보너스를 줘 행동을 강화한다.
# 감지(추측 금지, RAM 측정): 직전 스텝 마리오 앞 STOMP_DETECT_PX 이내에 있던 활성 적 슬롯이
# 이번 스텝에 비활성(0x0F~0x13=0)으로 바뀌고 마리오가 생존하면 = 밟아서 처치한 것.
GOOMBA_STOMP_BONUS = 50.0   # 밟기 1회 보너스 (사망 -100의 절반). [DAY3 트라이2] +150 실험했으나 밟기 비율 무변화로 +50 원복
STOMP_DETECT_PX = 48        # 직전 스텝 마리오 앞 이 거리(0~48px) 이내 적을 밟기 판정 대상으로

# --- 구덩이(낙사 함정) 거리 탐지 — [DAY4 트라이1] 바닥 타일맵 직접 읽기 ----------
# 굼바는 적 슬롯 좌표가 있지만 구덩이는 "객체"가 없다. 대신 SMB 플레이영역 타일맵(RAM 0x0500~)을
# 읽어, 마리오 앞 지면 칸이 비어 있으면(타일 0) 거기를 구덩이로 본다. (추측 금지 — [pit] 로그로 검증)
#
# SMB 타일맵: 0x0500부터 2화면분(각 13행×16열=0xD0). 전역 픽셀(x)·타일 행(row 0~12)→주소:
#   page = (x // 256) % 2,  col = (x % 256)//16,  addr = 0x500 + page*0xD0 + row*16 + col
TILEMAP_BASE = 0x0500
TILE_PAGE_SIZE = 0xD0        # 한 화면 = 13행 × 16열 = 208바이트
TILE_ROWS = 13              # 타일 행 수(0~12)
GROUND_ROW = 12            # 평지 지면 행(맨 아래). [pit] 검증으로 확정(행11·12 동일하게 지면).
PIT_SCAN_TILES = 4         # 마리오 앞 몇 칸(16px)까지 구덩이를 살필지 (≤48px=NEAR까지면 충분)

# 구덩이 거리 3단계(굼바보다 한 단계 적게 — 상태 폭증 방지). 0 없음 / 1 가까움 / 2 코앞.
# 굼바와 달리 구덩이 점프는 "코앞이냐"가 핵심이라 3단계로 충분.
PIT_DIST_DANGER = 16        # 단계2 코앞: 마리오 앞 0~16px (지금 점프해야 건넘)
PIT_DIST_NEAR = 48          # 단계1 가까움: 16~48px (그 너머는 0 없음)

# --- [DAY4 트라이2] 구덩이 통과 보너스 -------------------------------------
# 구덩이를 무사히 건너면 보너스. 굼바 밟기(detect_stomp)와 대칭 — 앞에 나타난 구덩이의 절대 x를
# 기억해, 마리오가 그 너머(폭+여유)까지 가고도 생존하면 "통과"로 본다. 빠지면 done(dead_pit)이라
# 보너스가 안 들어감 → 넘은 경우만 보상. (추측 금지 — 임시 [clear] 로그로 검증)
PIT_CLEAR_BONUS = 50.0       # 통과 1회 보너스 (굼바 밟기와 통일, 사망 -100의 절반)
PIT_CLEAR_MARGIN = 32        # 구덩이 시작 x로부터 이만큼 더 가야(폭 너머) 통과로 인정


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


def step_with_skip(env, action):
    """행동을 FRAME_SKIP 프레임 동안 유지하며 진행한다. 중간에 done 이면 즉시 멈춘다.

    매 프레임 화면을 갱신(스트리밍 부드럽게)하되, 소켓 왕복은 이 함수 1회당 1번만 일어나므로
    한 판이 약 FRAME_SKIP 배 빨리 끝난다. 보상은 서버가 x_pos 변화로 직접 계산하므로
    (compute_reward) 여기서 누적하지 않고, 스킵 후의 최종 obs/info 만 반환한다.
    """
    obs = info = None
    done = False
    for _ in range(max(1, FRAME_SKIP)):
        obs, _, done, info = step_compat(env, action)
        render_safe(env)
        set_frame(obs)
        if done:
            break
    return obs, done, info


def get_ram(env):
    """JoypadSpace 래퍼 안쪽 nes-py 환경의 RAM(2KB numpy 배열)을 best-effort 로 얻는다.

    JoypadSpace(gym.Wrapper)는 .ram 을 직접 노출하지 않을 수 있어 .unwrapped 까지 확인한다.
    """
    ram = getattr(env, "ram", None)
    if ram is None:
        ram = getattr(getattr(env, "unwrapped", env), "ram", None)
    return ram


def nearest_enemy_ahead(env, mario_x):
    """마리오 앞쪽(오른쪽)에서 가장 가까운 활성 적까지의 거리(px)를 반환한다. 없으면 None.

    RAM의 적 슬롯 5개를 훑어 활성(ADDR_ENEMY_DRAWN≠0) 슬롯의 전역 x 를 구하고,
    마리오보다 앞(dx≥0)인 것 중 최솟값을 고른다. (이미 지나친 뒤쪽 적은 무시)
    """
    ram = get_ram(env)
    if ram is None:
        return None
    nearest = None
    for i in range(ENEMY_SLOTS):
        if int(ram[ADDR_ENEMY_DRAWN + i]) == 0:        # 빈 슬롯
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + i]) * 256 + int(ram[ADDR_ENEMY_X + i])
        dx = enemy_x - mario_x
        if dx < 0:                                     # 뒤쪽(지나친) 적
            continue
        if nearest is None or dx < nearest:
            nearest = dx
    return nearest


def detect_enemy_distance(env, mario_x):
    """마리오 앞쪽 가장 가까운 적까지의 거리를 4단계로 매핑한다.

    0 없음 / 1 멂 / 2 가까움 / 3 위험함. (트라이3: 픽셀이 아니라 RAM 좌표로 계산)
    """
    nearest = nearest_enemy_ahead(env, mario_x)
    if nearest is None or nearest > ENEMY_DIST_FAR:
        return 0
    if nearest <= ENEMY_DIST_DANGER:
        return 3
    if nearest <= ENEMY_DIST_NEAR:
        return 2
    return 1


def read_tile(ram, x_global, row):
    """전역 픽셀 x와 타일 행(0~12)의 타일 값을 읽는다. (0=빈칸, 그 외=블록/지면)

    page = (x//256)%2 로 두 화면 중 어느 쪽인지 고르고, 화면 내 열·행으로 주소를 만든다.
    """
    if row < 0 or row >= TILE_ROWS:
        return 0
    page = (x_global // 256) % 2
    col = (x_global % 256) // 16
    return int(ram[TILEMAP_BASE + page * TILE_PAGE_SIZE + row * 16 + col])


def nearest_pit_ahead(env, mario_x):
    """마리오 앞쪽 지면 행(GROUND_ROW)이 비어 있는 가장 가까운 칸까지의 거리(px)를 반환. 없으면 None.

    굼바 nearest_enemy_ahead 와 대칭 — 앞 칸을 16px 단위로 훑어 첫 빈 지면(구덩이)을 찾는다.
    """
    ram = get_ram(env)
    if ram is None:
        return None
    for k in range(1, PIT_SCAN_TILES + 1):
        if read_tile(ram, mario_x + k * 16, GROUND_ROW) == 0:   # 지면 칸이 비었다 = 구덩이
            return k * 16
    return None


def detect_pit_distance(env, mario_x):
    """마리오 앞 가장 가까운 구덩이까지 거리를 3단계로 매핑한다. 0 없음 / 1 가까움 / 2 코앞.

    굼바 detect_enemy_distance 와 대칭. 단계만 4→3 (구덩이는 "코앞이냐"가 핵심).
    """
    nearest = nearest_pit_ahead(env, mario_x)
    if nearest is None or nearest > PIT_DIST_NEAR:
        return 0
    if nearest <= PIT_DIST_DANGER:
        return 2
    return 1


def active_enemies_ahead(env, mario_x):
    """마리오 앞 STOMP_DETECT_PX 이내에 있는 활성 적 슬롯 인덱스 집합을 반환한다.

    밟기 판정의 '직전 상태'로 쓴다 — 이 슬롯이 다음 스텝에 비활성(0)이 되면 밟힌 것으로 본다.
    (거리 4단계 detect_enemy_distance 와 달리, 슬롯 인덱스를 추적해야 사라짐을 잡을 수 있어 별도 함수)
    """
    ram = get_ram(env)
    slots = set()
    if ram is None:
        return slots
    for i in range(ENEMY_SLOTS):
        if int(ram[ADDR_ENEMY_DRAWN + i]) == 0:        # 빈 슬롯
            continue
        enemy_x = int(ram[ADDR_ENEMY_X_PAGE + i]) * 256 + int(ram[ADDR_ENEMY_X + i])
        dx = enemy_x - mario_x
        if 0 <= dx <= STOMP_DETECT_PX:                  # 마리오 앞 가까이만
            slots.add(i)
    return slots


def detect_stomp(env, prev_slots, done):
    """직전 스텝 마리오 앞에 있던 적 슬롯이 이번 스텝에 사라졌고(밟혀 처치) 마리오가 생존이면 True.

    - prev_slots: 직전 스텝의 active_enemies_ahead 결과(슬롯 인덱스 집합).
    - done(=마리오 사망/종료)이면 밟기 아님(부딪혀 죽은 경우 굼바는 살아있다).
    - 뛰어넘기는 굼바 슬롯이 active 인 채 뒤로 가므로 비활성화로 잡히지 않는다 → 밟기만 검출.
    """
    if done or not prev_slots:
        return False
    ram = get_ram(env)
    if ram is None:
        return False
    for i in prev_slots:
        if int(ram[ADDR_ENEMY_DRAWN + i]) == 0:        # 그 슬롯이 비워짐 = 밟혀 사라짐
            return True
    return False


def is_pit_death(info):
    """사망이 '낙사(구덩이 추락)'인지 대략 판단한다. (적 충돌과 구분하기 위함)

    마리오가 구덩이로 떨어지면 화면 아래로 내려가 y_pos 가 커진다. 이 값이
    PIT_Y_THRESHOLD 이상이면 낙사로 본다. (휴리스틱 — 첫 실행 로그로 임계 튜닝)
    """
    return int(info.get("y_pos", 0)) >= PIT_Y_THRESHOLD


def compute_reward(prev_x, info, done):
    """docs/03 의 보상표대로 보상과 종료 원인 문자열을 계산한다.

    [DAY2 트라이1] 사망을 낙사(dead_pit)/적(dead_enemy)으로 구분하고, 낙사만 페널티를 키운다.
    """
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
        elif is_pit_death(info):           # 낙사(구덩이 추락) — 페널티 키움(대조군)
            reward = PIT_DEATH_PENALTY
            status = "dead_pit"
        else:                              # 그 외 사망(적 충돌 등)
            reward = DEATH_PENALTY
            status = "dead_enemy"
    return reward, status


def build_state(info, reward, done, status, env):
    """Java GameState 모델과 같은 필드의 dict 를 만든다 (snake_case)."""
    mario_x = int(info.get("x_pos", 0))
    return {
        "mario_x": mario_x,
        "mario_y": int(info.get("y_pos", 0)),
        "enemy_dist": detect_enemy_distance(env, mario_x),
        "pit_dist": detect_pit_distance(env, mario_x),
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

    episode = 0
    while True:  # 에피소드 무한 반복 (항상 1-1)
        episode += 1
        reset_result = env.reset()
        _obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        # reset 직후에는 info(x_pos 등)가 없으므로 NOOP 한 스텝으로 초기 정보를 얻는다.
        obs, _, done, info = step_compat(env, 0)
        render_safe(env)
        set_frame(obs)
        prev_x = int(info.get("x_pos", 0))
        max_x = prev_x
        status = "running"
        prev_slots = active_enemies_ahead(env, prev_x)  # [DAY3] 밟기 판정용 직전 적 슬롯
        stomp_count = 0                                 # [DAY3] 그 판에서 밟은 횟수
        pending_pit_x = None                            # [DAY4 트라이2] 추적 중인 구덩이 시작 절대 x
        pit_clear_count = 0                             # [DAY4 트라이2] 그 판에서 통과한 구덩이 수

        # ① 첫 상태 전송 (아직 Java 행동 전 — 보상 0, done False)
        send_state(build_state(info, 0.0, False, "running", env))

        while not done:
            # ② Java 행동 수신
            line = rfile.readline()
            if not line:
                raise ConnectionError("Java 클라이언트 연결이 종료되었습니다.")
            action = int(json.loads(line)["action"])

            # ③ 게임을 FRAME_SKIP 프레임 진행 (같은 행동 유지 → 소켓 왕복 1/FRAME_SKIP)
            obs, done, info = step_with_skip(env, action)

            reward, status = compute_reward(prev_x, info, done)

            # [DAY3 트라이1] 밟기 보너스 — 직전 적 슬롯이 사라졌고 생존이면 처치로 보고 보너스.
            if detect_stomp(env, prev_slots, done):
                reward += GOOMBA_STOMP_BONUS
                stomp_count += 1

            prev_x = int(info.get("x_pos", prev_x))
            prev_slots = active_enemies_ahead(env, prev_x)
            max_x = max(max_x, prev_x)

            # [DAY4 트라이2] 구덩이 통과 보너스 — 앞에 구덩이가 나타나면 그 절대 x를 기억하고,
            # 마리오가 폭 너머(+MARGIN)까지 가고도 생존(not done)이면 건넌 것으로 보고 보너스.
            pit_px = nearest_pit_ahead(env, prev_x)
            if pit_px is not None and pit_px <= PIT_DIST_NEAR and pending_pit_x is None:
                pending_pit_x = prev_x + pit_px                       # 추적 시작(구덩이 시작 x)
            if pending_pit_x is not None and not done and prev_x > pending_pit_x + PIT_CLEAR_MARGIN:
                reward += PIT_CLEAR_BONUS
                pit_clear_count += 1
                pending_pit_x = None

            # ④ 결과 상태 전송 (done 이면 종료 신호)
            state = build_state(info, reward, done, status, env)
            send_state(state)

        # [DAY2 트라이1] 종료 원인 로그 — 낙사율 집계 + y_pos 임계(PIT_Y_THRESHOLD) 실측/튜닝용.
        # [DAY3 트라이1] stomps=N(그 판 밟기 횟수) 추가 — 통과율과 함께 행동 강화 효과 측정용.
        print(f"[Ep {episode:4d}] status={status:10s} maxX={max_x:5d} "
              f"endX={prev_x:5d} endY={int(info.get('y_pos', 0)):3d} "
              f"stomps={stomp_count} pitclears={pit_clear_count}", flush=True)
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
