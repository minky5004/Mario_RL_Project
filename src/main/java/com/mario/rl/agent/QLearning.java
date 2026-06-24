package com.mario.rl.agent;

import java.util.Random;

/**
 * Q-Learning — <b>off-policy</b> TD. [DAY10에서 {@link TabularBrain} 베이스로 분리]
 *
 * <p>다음 상태를 그 상태에서 <b>가장 좋은</b> 행동의 값으로 부트스트랩한다(실제 다음 행동은 무시).
 * "최선을 가정하고 배운다" — 탐험으로 무슨 행동을 했든 학습 목표는 max다.</p>
 *
 * <pre>target = reward + γ * max_a( qCommon[nc][a] + qLocal[nl][a] )</pre>
 *
 * <p>알고리즘 본체는 부트스트랩 한 줄뿐이고, 두 테이블·선택·갱신·저장은 모두 {@link TabularBrain}에 있다.</p>
 */
public class QLearning extends TabularBrain {

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public QLearning() {
        super(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public QLearning(long seed) {
        super(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public QLearning(long seed, int activeActions) {
        super(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public QLearning(int activeActions) {
        super(new Random(), activeActions);
    }

    /** off-policy: 다음 상태 행동들의 최댓값(다음 행동 무시). */
    @Override
    protected double bootstrap(int nextCommon, int nextLocal, int nextAction) {
        return maxCombined(nextCommon, nextLocal);
    }
}
