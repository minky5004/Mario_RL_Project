package com.mario.rl.agent;

import com.mario.rl.model.Action;

import java.util.Random;

/**
 * 랜덤 기준선 — <b>학습하지 않는</b> 두뇌. [DAY10 트라이2]
 *
 * <p>상태를 보지 않고 매 스텝 균등 무작위로 행동을 고른다. Q-Table도, 학습도, epsilon 감쇠도 없다.
 * 다른 알고리즘을 견주는 <b>바닥선</b>이다: <i>"학습한 두뇌라면 적어도 이것보단 나아야 한다."</i>
 * Q-Learning·SARSA의 성적이 <i>분산 위에서</i> 진짜로 학습한 결과인지(아니면 분산 그 자체인지)를 가른다.</p>
 *
 * <p>{@link TabularBrain}을 상속하지 않는다 — 테이블도 부트스트랩도 없으므로 {@link Brain}을 직접 구현한다.
 * epsilon은 개념상 항상 1.0(100% 탐험)이라 로그에 그대로 보고한다. 저장/로드 대상도 아니다
 * ({@code Main}이 {@link TabularBrain}만 저장/로드).</p>
 */
public class RandomBrain implements Brain {

    /** 난수 생성기. */
    private final Random random;
    /**
     * 고를 행동 수 (≤ {@link Action#ACTION_SIZE}). [DAY9 옵션 호환] 6=긴 점프 OFF, 7=포함.
     * 다른 두뇌와 같은 행동 공간에서 비교하기 위해 동일하게 받는다.
     */
    private final int activeActions;

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public RandomBrain() {
        this(new Random(), Action.ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public RandomBrain(long seed) {
        this(new Random(seed), Action.ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public RandomBrain(long seed, int activeActions) {
        this(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public RandomBrain(int activeActions) {
        this(new Random(), activeActions);
    }

    private RandomBrain(Random random, int activeActions) {
        if (activeActions < 1 || activeActions > Action.ACTION_SIZE) {
            throw new IllegalArgumentException(
                    "activeActions 는 1~" + Action.ACTION_SIZE + " 범위여야 함: " + activeActions);
        }
        this.random = random;
        this.activeActions = activeActions;
    }

    /** 상태를 무시하고 균등 무작위로 고른다. */
    @Override
    public int selectAction(int common, int local) {
        return random.nextInt(activeActions);
    }

    /** 기준선은 배우지 않는다(no-op). */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        // 학습 없음.
    }

    /** epsilon 감쇠 없음(항상 무작위, no-op). */
    @Override
    public void endEpisode() {
        // 고정.
    }

    /** 항상 1.0 — 100% 탐험(로그·그래프용). */
    @Override
    public double getEpsilon() {
        return 1.0;
    }
}
