package com.mario.rl.agent;

import java.util.Random;

/**
 * SARSA — <b>on-policy</b> TD. [DAY10 트라이1]
 *
 * <p>다음 상태를 그 상태에서 <b>실제로 고른</b> 행동의 값으로 부트스트랩한다(이름의 어원:
 * <b>S</b>tate–<b>A</b>ction–<b>R</b>eward–<b>S</b>tate–<b>A</b>ction). "내가 진짜 한 대로 배운다" —
 * 탐험(랜덤)으로 위험한 행동을 했다면 그 위험이 학습 목표에 반영돼, Q-Learning보다 보수적인 정책이 된다.</p>
 *
 * <pre>target = reward + γ * ( qCommon[nc][na] + qLocal[nl][na] )   // na = 실제 다음 행동</pre>
 *
 * <p>Q-Learning과 <b>부트스트랩 한 줄만</b> 다르다(max → 실제 다음 행동). 나머지는 모두 {@link TabularBrain}.</p>
 */
public class SARSA extends TabularBrain {

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public SARSA() {
        super(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public SARSA(long seed) {
        super(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public SARSA(long seed, int activeActions) {
        super(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public SARSA(int activeActions) {
        super(new Random(), activeActions);
    }

    /** on-policy: 다음 상태에서 실제로 고른 행동의 값. */
    @Override
    protected double bootstrap(int nextCommon, int nextLocal, int nextAction) {
        return combined(nextCommon, nextLocal, nextAction);
    }
}
