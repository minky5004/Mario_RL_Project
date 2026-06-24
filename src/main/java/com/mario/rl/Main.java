package com.mario.rl;

import com.mario.rl.agent.QLearning;
import com.mario.rl.agent.RLAgent;
import com.mario.rl.network.SocketClient;
import com.mario.rl.util.Logger;
import com.mario.rl.util.StateEncoder;

import java.io.IOException;

/**
 * 프로그램 진입점.
 *
 * <p>Python 게임 서버에 연결한 뒤, Q-Learning 에이전트를 만들어 정해진 에피소드 수만큼 학습한다.
 * 소켓은 try-with-resources로 열어 종료 시 자동으로 닫는다.</p>
 *
 * <p>실행 전 Python 서버({@code python/mario_env_server.py})가 먼저 떠 있어야 한다.
 * (연결이 늦어지면 {@link SocketClient}가 자동으로 재시도한다.)</p>
 */
public class Main {

    /** Python 서버 호스트. */
    private static final String HOST = "localhost";
    /** Python 서버 기본 포트. [DAY9] {@code -Dmario.port} 로 덮어쓸 수 있다(시드 병렬 실행용). */
    private static final int DEFAULT_PORT = 9999;
    /** 학습할 총 에피소드 수. [DAY6 트라이1] 500 → 1500 (예산 늘려 학습 희박 해소 검증). */
    private static final int MAX_EPISODES = 1500;

    /**
     * 프로그램 진입점.
     *
     * @param args 사용하지 않음
     */
    public static void main(String[] args) {
        System.out.println("=== Mario RL (Q-Learning) 학습 시작 ===");

        // [DAY9] 실행 옵션 — 시스템 프로퍼티로 전달(기본은 기존 동작 = 시드 없음·로드/저장 없음, 회귀 방지).
        //   -Dmario.seed=N   : 난수 시드 고정(시드 N회 반복 비교용, 작업1)
        //   -Dmario.load=경로 : 시작 시 Q-Table 로드(이어학습, 작업3)
        //   -Dmario.save=경로 : 학습 종료 시 Q-Table 저장(작업3)
        Long seed = parseLongProperty("mario.seed");
        String loadPath = System.getProperty("mario.load");
        String savePath = System.getProperty("mario.save");
        Long portProp = parseLongProperty("mario.port");
        int port = (portProp != null) ? portProp.intValue() : DEFAULT_PORT;
        // [DAY9 트라이1 베이스라인] 쓰는 행동 수. 미지정/7=긴 점프 포함, 6=긴 점프 끔(DAY8 매칭).
        Long actionsProp = parseLongProperty("mario.actions");

        try (SocketClient socketClient = new SocketClient(HOST, port)) {
            QLearning qLearning;
            if (seed != null && actionsProp != null) {
                qLearning = new QLearning(seed, actionsProp.intValue());
            } else if (seed != null) {
                qLearning = new QLearning(seed);
            } else {
                qLearning = new QLearning();
            }
            if (seed != null) {
                System.out.println("[Main] 시드 고정: " + seed);
            }
            if (actionsProp != null) {
                System.out.println("[Main] 행동 수 제한: " + actionsProp + " (긴 점프 " + (actionsProp >= 7 ? "포함" : "제외") + ")");
            }
            if (loadPath != null) {
                qLearning.load(loadPath);
                System.out.println("[Main] Q-Table 로드: " + loadPath);
            }
            StateEncoder stateEncoder = new StateEncoder();
            Logger logger = new Logger();

            RLAgent agent = new RLAgent(qLearning, socketClient, stateEncoder, logger);
            agent.train(MAX_EPISODES);

            if (savePath != null) {
                qLearning.save(savePath);
                System.out.println("[Main] Q-Table 저장: " + savePath);
            }
            System.out.println("=== 학습 종료 ===");
        } catch (IOException e) {
            System.err.println("[Main] 통신 오류로 종료: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("[Main] 연결 대기 중 중단됨: " + e.getMessage());
        }
    }

    /** 시스템 프로퍼티를 {@code Long}으로 파싱한다. 없거나 숫자가 아니면 {@code null}. */
    private static Long parseLongProperty(String key) {
        String value = System.getProperty(key);
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            System.err.println("[Main] " + key + " 값이 숫자가 아님(무시): " + value);
            return null;
        }
    }
}
