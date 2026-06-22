package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.model.GameState;
import com.mario.rl.network.SocketClient;
import com.mario.rl.util.Logger;
import com.mario.rl.util.StateEncoder;

import java.io.IOException;

/**
 * 강화학습 에이전트. 학습 루프 전체를 지휘한다.
 *
 * <p>{@link QLearning}(두뇌), {@link SocketClient}(통신), {@link StateEncoder}(상태 변환),
 * {@link Logger}(출력)를 주입받아, 에피소드마다 다음 순서를 반복한다:</p>
 *
 * <ol>
 *   <li>상태 수신 → 인코딩</li>
 *   <li>행동 선택({@link QLearning#selectAction}) → 전송</li>
 *   <li>다음 상태·보상 수신 → 인코딩</li>
 *   <li>Q-Table 갱신({@link QLearning#update})</li>
 *   <li>{@code done}이면 에피소드 종료 → epsilon 감쇠 → 로그 출력</li>
 * </ol>
 *
 * <p>통신 순서(Python 전송 → Java 수신 → Java 전송 → Python 수신)를 엄격히 지켜 데드락을 방지한다.</p>
 */
public class RLAgent {

    private final QLearning qLearning;
    private final SocketClient socketClient;
    private final StateEncoder stateEncoder;
    private final Logger logger;

    /**
     * 학습에 필요한 협력 객체들을 주입받아 에이전트를 생성한다.
     *
     * @param qLearning    Q-Learning 알고리즘
     * @param socketClient Python 서버와의 통신 클라이언트
     * @param stateEncoder 상태 인코더
     * @param logger       학습 로거
     */
    public RLAgent(QLearning qLearning, SocketClient socketClient,
                   StateEncoder stateEncoder, Logger logger) {
        this.qLearning = qLearning;
        this.socketClient = socketClient;
        this.stateEncoder = stateEncoder;
        this.logger = logger;
    }

    /**
     * 지정한 에피소드 수만큼 학습을 수행한다.
     *
     * @param maxEpisodes 학습할 에피소드 수
     * @throws IOException 통신 중 오류가 발생한 경우
     */
    public void train(int maxEpisodes) throws IOException {
        for (int episode = 1; episode <= maxEpisodes; episode++) {
            runEpisode(episode);
        }
    }

    /** 한 에피소드를 끝까지 진행하며 매 스텝 Q-Table을 갱신한다. */
    private void runEpisode(int episode) throws IOException {
        // 에피소드 첫 상태 수신 — [DAY7 트라이2] 공통/위치 두 인덱스로 인코딩
        GameState current = socketClient.receiveGameState();
        int common = stateEncoder.encodeCommon(current);
        int local = stateEncoder.encodeLocal(current);

        double totalReward = 0.0;
        int maxX = current.getMarioX();
        int step = 0;

        while (true) {
            // 1) 행동 선택 후 전송 (두 테이블 합산 기준)
            int actionIndex = qLearning.selectAction(common, local);
            socketClient.sendAction(Action.fromValue(actionIndex));

            // 2) 행동의 결과(다음 상태·보상·종료) 수신
            GameState next = socketClient.receiveGameState();
            int nextCommon = stateEncoder.encodeCommon(next);
            int nextLocal = stateEncoder.encodeLocal(next);
            double reward = next.getReward();
            boolean done = next.isDone();

            // 3) 두 Q-Table 동시 갱신 (학습)
            qLearning.update(common, local, actionIndex, reward, nextCommon, nextLocal, done);

            // 4) 통계 누적
            totalReward += reward;
            maxX = Math.max(maxX, next.getMarioX());
            logger.logStep(step, Action.fromValue(actionIndex), reward);

            // 5) 다음 스텝 준비
            common = nextCommon;
            local = nextLocal;
            step++;

            if (done) {
                break;
            }
        }

        // 에피소드 종료: 탐험 비율 감소 후 요약 로그
        qLearning.decayEpsilon();
        logger.logEpisode(episode, totalReward, maxX, qLearning.getEpsilon());
    }
}
