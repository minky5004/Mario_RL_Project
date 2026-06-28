package com.mario.rl;

import com.mario.rl.agent.Brain;
import com.mario.rl.agent.ExpectedSARSA;
import com.mario.rl.agent.GeneticAlgorithm;
import com.mario.rl.agent.MonteCarlo;
import com.mario.rl.agent.QLearning;
import com.mario.rl.agent.RLAgent;
import com.mario.rl.agent.RandomBrain;
import com.mario.rl.agent.SARSA;
import com.mario.rl.agent.TabularBrain;
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

        // [DAY9~10] 실행 옵션 — 시스템 프로퍼티(미지정이면 기존 동작 = qlearning·시드 없음·로드/저장 없음, 회귀 방지).
        //   -Dmario.algo=qlearning|sarsa|expected_sarsa|monte_carlo|random|ga : 알고리즘 선택 [DAY10~13] (random=무학습 기준선, ga=유전 알고리즘)
        //   -Dmario.seed=N   : 난수 시드 고정(시드 N회 반복 비교용)
        //   -Dmario.port=N   : 서버 포트(시드 병렬 실행용)
        //   -Dmario.actions=N: 쓰는 행동 수(6=긴 점프 끔)
        //   -Dmario.load/save=경로 : Q-Table 로드/저장(표 알고리즘만)
        String algo = System.getProperty("mario.algo", "qlearning").trim().toLowerCase();
        Long seed = parseLongProperty("mario.seed");
        String loadPath = System.getProperty("mario.load");
        String savePath = System.getProperty("mario.save");
        Long portProp = parseLongProperty("mario.port");
        int port = (portProp != null) ? portProp.intValue() : DEFAULT_PORT;
        Long actionsProp = parseLongProperty("mario.actions");

        try (SocketClient socketClient = new SocketClient(HOST, port)) {
            Brain brain = createBrain(algo, seed, actionsProp);
            System.out.println("[Main] 알고리즘: " + algo);
            if (seed != null) {
                System.out.println("[Main] 시드 고정: " + seed);
            }
            if (actionsProp != null) {
                System.out.println("[Main] 행동 수 제한: " + actionsProp + " (긴 점프 " + (actionsProp >= 7 ? "포함" : "제외") + ")");
            }
            // 저장/로드는 표(테이블) 알고리즘에서만 지원.
            if (loadPath != null && brain instanceof TabularBrain tb) {
                tb.load(loadPath);
                System.out.println("[Main] Q-Table 로드: " + loadPath);
            }
            StateEncoder stateEncoder = new StateEncoder();
            Logger logger = new Logger();

            RLAgent agent = new RLAgent(brain, socketClient, stateEncoder, logger);
            agent.train(MAX_EPISODES);

            if (savePath != null && brain instanceof TabularBrain tb) {
                tb.save(savePath);
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

    /**
     * 알고리즘 이름으로 두뇌를 만든다. [DAY10] seed·actions 조합을 알맞은 생성자로 라우팅.
     *
     * @param algo    "qlearning" | "sarsa" | "expected_sarsa" | "monte_carlo" | "random" | "ga" (그 외는 qlearning 으로 폴백)
     * @param seed    시드(null이면 비결정적)
     * @param actions 쓰는 행동 수(null이면 전체)
     * @return 생성된 {@link Brain}
     */
    private static Brain createBrain(String algo, Long seed, Long actions) {
        boolean hasSeed = seed != null;
        boolean hasActions = actions != null;
        long s = hasSeed ? seed : 0L;
        int a = hasActions ? actions.intValue() : 0;
        switch (algo) {
            case "sarsa":
                if (hasSeed && hasActions) return new SARSA(s, a);
                if (hasSeed) return new SARSA(s);
                if (hasActions) return new SARSA(a);
                return new SARSA();
            case "expected_sarsa":
                if (hasSeed && hasActions) return new ExpectedSARSA(s, a);
                if (hasSeed) return new ExpectedSARSA(s);
                if (hasActions) return new ExpectedSARSA(a);
                return new ExpectedSARSA();
            case "monte_carlo":
                if (hasSeed && hasActions) return new MonteCarlo(s, a);
                if (hasSeed) return new MonteCarlo(s);
                if (hasActions) return new MonteCarlo(a);
                return new MonteCarlo();
            case "random":
                if (hasSeed && hasActions) return new RandomBrain(s, a);
                if (hasSeed) return new RandomBrain(s);
                if (hasActions) return new RandomBrain(a);
                return new RandomBrain();
            case "ga":
            case "genetic":
                if (hasSeed && hasActions) return new GeneticAlgorithm(s, a);
                if (hasSeed) return new GeneticAlgorithm(s);
                if (hasActions) return new GeneticAlgorithm(a);
                return new GeneticAlgorithm();
            case "qlearning":
            default:
                if (hasSeed && hasActions) return new QLearning(s, a);
                if (hasSeed) return new QLearning(s);
                if (hasActions) return new QLearning(a);
                return new QLearning();
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
