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
    /** Python 서버 포트. */
    private static final int PORT = 9999;
    /** 학습할 총 에피소드 수. [DAY6 트라이1] 500 → 1500 (예산 늘려 학습 희박 해소 검증). */
    private static final int MAX_EPISODES = 1500;

    /**
     * 프로그램 진입점.
     *
     * @param args 사용하지 않음
     */
    public static void main(String[] args) {
        System.out.println("=== Mario RL (Q-Learning) 학습 시작 ===");

        try (SocketClient socketClient = new SocketClient(HOST, PORT)) {
            QLearning qLearning = new QLearning();
            StateEncoder stateEncoder = new StateEncoder();
            Logger logger = new Logger();

            RLAgent agent = new RLAgent(qLearning, socketClient, stateEncoder, logger);
            agent.train(MAX_EPISODES);

            System.out.println("=== 학습 종료 ===");
        } catch (IOException e) {
            System.err.println("[Main] 통신 오류로 종료: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("[Main] 연결 대기 중 중단됨: " + e.getMessage());
        }
    }
}
