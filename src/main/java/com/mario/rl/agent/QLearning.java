package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.util.Random;

/**
 * Q-Learning 핵심 알고리즘.
 *
 * <p>{@code double[STATE_SIZE][ACTION_SIZE]} 크기의 Q-Table을 유지하며, epsilon-greedy 정책으로
 * 행동을 선택하고 시간차(TD) 갱신식으로 Q값을 학습한다.</p>
 *
 * <p>Q값 갱신식:
 * <pre>
 * target = done ? reward : reward + γ * max(Q[nextState])
 * Q[s][a] += α * (target - Q[s][a])
 * </pre></p>
 */
public class QLearning {

    /** 상태 차원 크기. */
    private static final int STATE_SIZE = StateEncoder.STATE_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 학습률 (α). */
    private static final double LEARNING_RATE = 0.1;
    /** 할인율 (γ). */
    private static final double DISCOUNT_FACTOR = 0.99;
    /** epsilon 초기값 (탐험 비율). */
    private static final double INITIAL_EPSILON = 1.0;
    /** 에피소드마다 곱해지는 epsilon 감쇠율. */
    private static final double EPSILON_DECAY = 0.995;
    /** epsilon 하한값. */
    private static final double EPSILON_MIN = 0.01;

    /** Q-Table: [상태][행동] = 기대 보상. */
    private final double[][] qTable;
    /** 난수 생성기 (탐험 및 무작위 동점 처리용). */
    private final Random random;
    /** 현재 탐험 비율. */
    private double epsilon;

    /**
     * Q-Learning 에이전트를 생성한다. Q-Table은 0으로 초기화되고 epsilon은 1.0에서 시작한다.
     */
    public QLearning() {
        this.qTable = new double[STATE_SIZE][ACTION_SIZE];
        this.random = new Random();
        this.epsilon = INITIAL_EPSILON;
    }

    /**
     * epsilon-greedy 정책으로 행동 인덱스를 선택한다.
     *
     * <p>확률 epsilon으로 무작위 행동(탐험), 그 외에는 현재 상태에서 Q값이 가장 큰 행동(활용)을 선택한다.</p>
     *
     * @param state 상태 인덱스 (0 ~ STATE_SIZE-1)
     * @return 선택된 행동 인덱스 (0 ~ ACTION_SIZE-1)
     */
    public int selectAction(int state) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(ACTION_SIZE);
        }
        return argMax(qTable[state]);
    }

    /**
     * Q-Learning 갱신식으로 Q-Table을 갱신한다.
     *
     * @param state     현재 상태 인덱스
     * @param action    수행한 행동 인덱스
     * @param reward    받은 보상
     * @param nextState 다음 상태 인덱스
     * @param done      에피소드 종료 여부 (true면 미래 보상을 더하지 않음)
     */
    public void update(int state, int action, double reward, int nextState, boolean done) {
        double target = done
                ? reward
                : reward + DISCOUNT_FACTOR * maxValue(qTable[nextState]);
        qTable[state][action] += LEARNING_RATE * (target - qTable[state][action]);
    }

    /**
     * epsilon을 감쇠시킨다. 하한값({@value #EPSILON_MIN}) 아래로는 내려가지 않는다.
     * 일반적으로 에피소드 종료 시 호출한다.
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

    /**
     * 현재 Q-Table을 반환한다 (저장 등 외부 활용용).
     *
     * @return Q-Table 2차원 배열
     */
    public double[][] getQTable() {
        return qTable;
    }

    /** 배열에서 최댓값의 인덱스를 반환한다. 동점이면 더 앞선 인덱스를 반환한다. */
    private int argMax(double[] values) {
        int bestIndex = 0;
        double bestValue = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    /** 배열의 최댓값을 반환한다. */
    private double maxValue(double[] values) {
        double max = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        return max;
    }
}
