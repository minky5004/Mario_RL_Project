package com.mario.rl.agent;

import java.util.Random;

/**
 * Expected SARSA — <b>on-policy</b> TD(기댓값판). [DAY10 트라이3]
 *
 * <p>SARSA는 "실제로 고른 다음 행동 하나"로 부트스트랩하는데, Expected SARSA는 그 자리에
 * <b>다음 상태에서 정책(ε-greedy)이 고를 행동들의 기댓값</b>을 쓴다. 즉 "다음에 운으로 뭘 골랐나"의
 * 분산을 정책 평균으로 지워, SARSA의 on-policy 성질은 유지하면서 갱신을 더 안정적으로 만든다.</p>
 *
 * <p>ε-greedy 정책에서 best 행동 하나는 {@code (1-ε)+ε/n}, 나머지는 각 {@code ε/n} 확률이므로
 * 기댓값이 깔끔하게 닫힌다(우리 {@code selectAction}은 동점 시 첫 best 하나만 greedy로 고르므로
 * {@code (1-ε)}는 그 하나에만 간다 — 정책과 정확히 일치):</p>
 *
 * <pre>
 * E[Q(s',·)] = Σ_a π(a) Q(s',a) = (1-ε)·max_a Q(s',a) + ε·mean_a Q(s',a)
 * target     = reward + γ * E[Q(s',·)]
 * </pre>
 *
 * <p>Q-Learning(max)·SARSA(실제 다음 행동)와 마찬가지로 <b>부트스트랩 한 줄만</b> 다르다.
 * ε이 0이면 max뿐이라 Q-Learning과 같아지고, ε이 클수록 평균 쪽으로 기운다.</p>
 */
public class ExpectedSARSA extends TabularBrain {

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public ExpectedSARSA() {
        super(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public ExpectedSARSA(long seed) {
        super(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public ExpectedSARSA(long seed, int activeActions) {
        super(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public ExpectedSARSA(int activeActions) {
        super(new Random(), activeActions);
    }

    /**
     * on-policy(기댓값): 다음 상태에서 ε-greedy 정책이 고를 행동의 기대 가치.
     * {@code (1-ε)·max + ε·mean} (실제 다음 행동 {@code nextAction}은 쓰지 않는다).
     */
    @Override
    protected double bootstrap(int nextCommon, int nextLocal, int nextAction) {
        double sum = 0.0;
        for (int a = 0; a < activeActions; a++) {
            sum += combined(nextCommon, nextLocal, a);
        }
        double mean = sum / activeActions;
        double max = maxCombined(nextCommon, nextLocal);
        return (1.0 - epsilon) * max + epsilon * mean;
    }
}
