package com.mario.rl.agent;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Monte Carlo — <b>부트스트랩 없는</b> 표 기반 학습. [DAY12]
 *
 * <p>TD 가족(Q-Learning·SARSA·Expected SARSA)은 매 스텝 "다음 상태의 추정치"로 현재를 갱신한다(부트스트랩).
 * Monte Carlo는 추정에 추정을 쌓지 않는다 — <b>에피소드가 끝날 때까지 기다렸다가</b>, 그 판에서 실제로 받은
 * 보상들의 할인 합(return G)으로 한 번에 갱신한다. "끝을 보고 나서 배운다."</p>
 *
 * <pre>
 * G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …        // 에피소드 끝에서 역방향으로 누적
 * target = G_t                                   // bootstrap 자리에 '실제 return'이 들어감
 * </pre>
 *
 * <p>구현은 <b>every-visit, constant-α</b> MC다 — 한 에피소드에서 같은 (상태,행동)을 여러 번 밟으면 매번 갱신하고,
 * 갱신은 {@link TabularBrain#updateToward}로 두 테이블에 α/2씩 당긴다(TD 가족과 같은 갱신식·같은 두 테이블).
 * 다른 점은 <b>타이밍 하나</b>: {@link #learn}은 갱신하지 않고 궤적만 쌓고, 진짜 갱신은 {@link #endEpisode}에서 한다.</p>
 *
 * <p>TD가 아니라 부트스트랩 개념이 없으므로 {@link #bootstrap}은 쓰지 않는다(호출되면 버그 → 예외).
 * {@link TabularBrain}의 "단 한 줄(bootstrap)만 다르다" 패턴을 처음 벗어난 두뇌다.</p>
 */
public class MonteCarlo extends TabularBrain {

    /** 한 에피소드 동안 모으는 궤적 — 스텝마다 (상태 두 인덱스, 행동, 받은 보상). */
    private record Step(int common, int local, int action, double reward) {
    }

    /** 현재 에피소드의 궤적. {@link #endEpisode}에서 역방향으로 소비하고 비운다. */
    private final List<Step> trajectory = new ArrayList<>();

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public MonteCarlo() {
        super(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public MonteCarlo(long seed) {
        super(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public MonteCarlo(long seed, int activeActions) {
        super(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public MonteCarlo(int activeActions) {
        super(new Random(), activeActions);
    }

    /**
     * 아직 갱신하지 않는다 — 이 스텝의 경험을 궤적에 쌓아만 둔다.
     * (return은 에피소드가 끝나야 알 수 있으므로 실제 학습은 {@link #endEpisode}에서.)
     * 부트스트랩이 없어 {@code next*}·{@code done}은 쓰지 않는다.
     */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        trajectory.add(new Step(common, local, action, reward));
    }

    /**
     * 에피소드 종료: 궤적을 <b>역방향</b>으로 훑어 return G를 누적하고, 매 스텝을 G쪽으로 갱신한다.
     * 그 뒤 epsilon 감쇠({@code super.endEpisode()})하고 궤적을 비운다.
     */
    @Override
    public void endEpisode() {
        double g = 0.0;
        for (int t = trajectory.size() - 1; t >= 0; t--) {
            Step s = trajectory.get(t);
            g = s.reward() + DISCOUNT_FACTOR * g;
            updateToward(s.common(), s.local(), s.action(), g);
        }
        trajectory.clear();
        super.endEpisode();
    }

    /** Monte Carlo는 부트스트랩하지 않는다 — 호출되면 학습 루프 버그다. */
    @Override
    protected double bootstrap(int nextCommon, int nextLocal, int nextAction) {
        throw new UnsupportedOperationException(
                "Monte Carlo는 부트스트랩을 쓰지 않는다 — 갱신은 endEpisode의 실제 return으로 한다.");
    }
}
