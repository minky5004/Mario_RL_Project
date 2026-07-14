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

    private final Brain brain;
    private final SocketClient socketClient;
    private final StateEncoder stateEncoder;
    private final Logger logger;

    /**
     * 학습에 필요한 협력 객체들을 주입받아 에이전트를 생성한다.
     *
     * @param brain        두뇌(알고리즘) — Q-Learning·SARSA 등 {@link Brain} 구현 [DAY10]
     * @param socketClient Python 서버와의 통신 클라이언트
     * @param stateEncoder 상태 인코더
     * @param logger       학습 로거
     */
    public RLAgent(Brain brain, SocketClient socketClient,
                   StateEncoder stateEncoder, Logger logger) {
        this.brain = brain;
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

    /**
     * 한 에피소드를 끝까지 진행하며 매 스텝 학습한다.
     *
     * <p>[DAY10] SARSA식 루프 — 다음 행동 {@code nextAction}을 학습 *전에* 골라 {@link Brain#learn}에 넘긴다.
     * on-policy(SARSA)는 그 값을, off-policy(Q-Learning)는 무시하고 max를 쓰므로 같은 루프로 둘 다 표현된다.</p>
     */
    private void runEpisode(int episode) throws IOException {
        // 에피소드 첫 상태 수신 — [DAY7 트라이2] 공통/위치 두 인덱스로 인코딩
        GameState current = socketClient.receiveGameState();
        int common = stateEncoder.encodeCommon(current);
        int local = stateEncoder.encodeLocal(current);
        int action = brain.selectAction(common, local);

        double totalReward = 0.0;
        int maxX = current.getMarioX();
        int step = 0;

        while (true) {
            // 1) 현재 행동 전송
            socketClient.sendAction(Action.fromValue(action));

            // 2) 행동의 결과(다음 상태·보상·종료) 수신
            GameState next = socketClient.receiveGameState();
            int nextCommon = stateEncoder.encodeCommon(next);
            int nextLocal = stateEncoder.encodeLocal(next);
            double reward = next.getReward();
            boolean done = next.isDone();

            // 3) 다음 행동을 먼저 고른다(SARSA 부트스트랩용). 종료면 의미 없음.
            int nextAction = done ? -1 : brain.selectAction(nextCommon, nextLocal);

            // 4) 학습 (알고리즘에 따라 nextAction 사용/무시)
            brain.learn(common, local, action, reward, nextCommon, nextLocal, nextAction, done);

            // 5) 통계 누적
            totalReward += reward;
            maxX = Math.max(maxX, next.getMarioX());
            logger.logStep(step, Action.fromValue(action), reward);

            // 6) 다음 스텝 준비
            common = nextCommon;
            local = nextLocal;
            action = nextAction;
            step++;

            if (done) {
                break;
            }
        }

        // 에피소드 종료: 결과(도달 거리) 통보 → 탐험 비율 감소 → 요약 로그
        // [DAY18] maxX는 여기(루프)에만 있으므로 두뇌에 따로 알린다. 쓰는 두뇌는 GA v2뿐(적합도=maxX).
        brain.observeEpisodeOutcome(maxX);
        brain.endEpisode();
        logger.logEpisode(episode, totalReward, maxX, brain.getEpsilon());
    }
}
