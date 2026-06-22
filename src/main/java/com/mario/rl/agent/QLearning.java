package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.util.Random;

/**
 * Q-Learning 핵심 알고리즘.
 *
 * <p>[DAY7 트라이2] <b>두 개의 Q-Table을 합산</b>해 행동을 고른다(tile/coarse coding):</p>
 * <ul>
 *   <li>{@code qCommon[COMMON_SIZE][A]} — 위치를 무시한 상태(`Q_공통`). 모든 위치의 경험이 합쳐져
 *       "굼바=넘기"를 빠르게 일반화한다.</li>
 *   <li>{@code qLocal[LOCAL_SIZE][A]} — 위치(xZone)까지 포함(`Q_위치`). 특정 지점의 예외를 덧칠해 안정화.</li>
 * </ul>
 *
 * <p>행동 가치는 두 테이블의 합 {@code Q(s,a) = qCommon[c][a] + qLocal[l][a]} 이다.
 * TD 오차 δ는 양쪽에 똑같이, 단 학습률을 타일 수(2)로 나눠 적용한다(합산 Q의 실효 학습률 발산 방지):
 * <pre>
 * target = done ? reward : reward + γ * max_a( qCommon[nc][a] + qLocal[nl][a] )
 * δ = target - (qCommon[c][a] + qLocal[l][a])
 * qCommon[c][a] += (α/2) * δ ;  qLocal[l][a] += (α/2) * δ
 * </pre></p>
 */
public class QLearning {

    /** `Q_공통`(위치 무시) 상태 크기. */
    private static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** `Q_위치`(위치 포함) 상태 크기. */
    private static final int LOCAL_SIZE = StateEncoder.LOCAL_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 학습률 (α). */
    private static final double LEARNING_RATE = 0.1;
    /** 테이블별 학습률 — TD 오차를 두 테이블(타일)에 나눠 적용(α/타일수). */
    private static final double TABLE_LR = LEARNING_RATE / 2.0;
    /** 할인율 (γ). */
    private static final double DISCOUNT_FACTOR = 0.99;
    /** epsilon 초기값 (탐험 비율). */
    private static final double INITIAL_EPSILON = 1.0;
    /** 에피소드마다 곱해지는 epsilon 감쇠율. */
    private static final double EPSILON_DECAY = 0.995;
    /** epsilon 하한값. */
    private static final double EPSILON_MIN = 0.01;

    /** `Q_공통` 테이블: [위치 무시 상태][행동]. */
    private final double[][] qCommon;
    /** `Q_위치` 테이블: [위치 포함 상태][행동]. */
    private final double[][] qLocal;
    /** 난수 생성기 (탐험 및 무작위 동점 처리용). */
    private final Random random;
    /** 현재 탐험 비율. */
    private double epsilon;

    /**
     * Q-Learning 에이전트를 생성한다. 두 Q-Table은 0으로 초기화되고 epsilon은 1.0에서 시작한다.
     */
    public QLearning() {
        this.qCommon = new double[COMMON_SIZE][ACTION_SIZE];
        this.qLocal = new double[LOCAL_SIZE][ACTION_SIZE];
        this.random = new Random();
        this.epsilon = INITIAL_EPSILON;
    }

    /**
     * epsilon-greedy 정책으로 행동 인덱스를 선택한다(두 테이블 합산 기준).
     *
     * @param common `Q_공통` 상태 인덱스
     * @param local  `Q_위치` 상태 인덱스
     * @return 선택된 행동 인덱스 (0 ~ ACTION_SIZE-1)
     */
    public int selectAction(int common, int local) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(ACTION_SIZE);
        }
        return argMaxCombined(common, local);
    }

    /**
     * 두 Q-Table을 동시에 갱신한다. TD 오차를 양쪽에 α/2 씩 적용한다.
     *
     * @param common     현재 `Q_공통` 인덱스
     * @param local      현재 `Q_위치` 인덱스
     * @param action     수행한 행동 인덱스
     * @param reward     받은 보상
     * @param nextCommon 다음 `Q_공통` 인덱스
     * @param nextLocal  다음 `Q_위치` 인덱스
     * @param done       에피소드 종료 여부 (true면 미래 보상을 더하지 않음)
     */
    public void update(int common, int local, int action, double reward,
                       int nextCommon, int nextLocal, boolean done) {
        double q = qCommon[common][action] + qLocal[local][action];
        double target = done
                ? reward
                : reward + DISCOUNT_FACTOR * maxCombined(nextCommon, nextLocal);
        double delta = target - q;
        qCommon[common][action] += TABLE_LR * delta;
        qLocal[local][action] += TABLE_LR * delta;
    }

    /**
     * epsilon을 감쇠시킨다. 하한값({@value #EPSILON_MIN}) 아래로는 내려가지 않는다.
     */
    public void decayEpsilon() {
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    /**
     * 현재 탐험 비율(epsilon)을 반환한다.
     *
     * @return 현재 epsilon 값
     */
    public double getEpsilon() {
        return epsilon;
    }

    /** 두 테이블 합산값이 최대인 행동 인덱스. 동점이면 더 앞선 인덱스. */
    private int argMaxCombined(int common, int local) {
        int bestIndex = 0;
        double bestValue = qCommon[common][0] + qLocal[local][0];
        for (int a = 1; a < ACTION_SIZE; a++) {
            double v = qCommon[common][a] + qLocal[local][a];
            if (v > bestValue) {
                bestValue = v;
                bestIndex = a;
            }
        }
        return bestIndex;
    }

    /** 두 테이블 합산값의 최댓값. */
    private double maxCombined(int common, int local) {
        double max = qCommon[common][0] + qLocal[local][0];
        for (int a = 1; a < ACTION_SIZE; a++) {
            double v = qCommon[common][a] + qLocal[local][a];
            if (v > max) {
                max = v;
            }
        }
        return max;
    }
}
