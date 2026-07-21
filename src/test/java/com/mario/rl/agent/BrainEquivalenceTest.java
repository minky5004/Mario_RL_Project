package com.mario.rl.agent;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * <b>두뇌 등가성(경계) 회귀 테스트.</b>
 *
 * <p>[DAY17]·[DAY20]에서 새 두뇌를 구현할 때마다 "손잡이를 끝까지 돌리면 이미 있는 두뇌와 같아지는가"를
 * 임시 클래스로 확인하고 지웠다(일지에 글로만 남음). 그 검증을 영구 테스트로 되살린다 —
 * <b>경계가 맞으면 중간값도 믿을 수 있다</b>는 게 당시 근거였으므로, 경계가 깨지면 실험 결과 전체가 흔들린다.</p>
 *
 * <ul>
 *   <li>[DAY17] {@link NStepSARSA} {@code n=1} ≡ {@link SARSA} — 창이 한 칸이면 보통의 1-step TD</li>
 *   <li>[DAY17] {@link NStepSARSA} {@code n=∞} ≡ {@link MonteCarlo} — 창이 안 차서 전부 꼬리(실제 return)로 마감</li>
 *   <li>[DAY20] {@link QLambda} {@code λ=0} ≡ {@link QLearning} — 흔적이 즉시 사라져 현재 칸만 갱신</li>
 *   <li>[DAY20] {@link QLambda}(항상 greedy) ≡ {@link SARSALambda} — Watkins cut이 한 번도 안 걸리면 둘은 같다</li>
 * </ul>
 *
 * <p>같은 경험 시퀀스를 두 두뇌에 먹이고 <b>두 Q-Table이 오차 0으로 일치</b>하는지 본다
 * (일지가 주장한 "오차 0"을 그대로 검사한다).</p>
 */
class BrainEquivalenceTest {

    /** 한 스텝의 경험 — 다음 상태는 시퀀스의 다음 원소에서 가져온다. */
    private record Step(int common, int local, int action, double reward) {
    }

    /** 재현 가능한 가짜 궤적을 만든다(상태·행동은 실제 인덱스 범위 안, 보상은 실제 보상 규칙에서 뽑음). */
    private static List<Step> trajectory(long seed, int length) {
        Random rng = new Random(seed);
        // 실제 보상 규칙(전진 +0.1×px / 후퇴 −5 / 사망 −100 / 클리어 +1000)에서 고른 값들
        double[] rewards = {0.0, 0.4, 1.2, 3.5, -5.0, -100.0, 1000.0};
        List<Step> steps = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            steps.add(new Step(
                    rng.nextInt(TabularBrain.COMMON_SIZE),
                    rng.nextInt(TabularBrain.LOCAL_SIZE),
                    rng.nextInt(TabularBrain.ACTION_SIZE),
                    rewards[rng.nextInt(rewards.length)]));
        }
        return steps;
    }

    /**
     * 궤적을 한 에피소드로 먹인다. 마지막 스텝은 {@code done=true}.
     *
     * @param greedyNext true면 다음 행동을 궤적 대신 <b>그 두뇌의 greedy 행동</b>으로 준다
     *                   (Watkins cut이 안 걸리는 상황을 만들기 위함)
     */
    private static void runEpisode(TabularBrain brain, List<Step> steps, boolean greedyNext) {
        for (int t = 0; t < steps.size(); t++) {
            Step s = steps.get(t);
            boolean done = (t == steps.size() - 1);
            Step next = done ? s : steps.get(t + 1);
            int nextAction = (!done && greedyNext)
                    ? brain.argMaxCombined(next.common(), next.local())
                    : next.action();
            brain.learn(s.common(), s.local(), s.action(), s.reward(),
                    next.common(), next.local(), nextAction, done);
        }
        brain.endEpisode();
    }

    /** 두 두뇌를 같은 궤적으로 여러 판 돌린다(에피소드 경계 처리까지 같은지 보려면 2판 이상이 필요). */
    private static void runBoth(TabularBrain a, TabularBrain b, boolean greedyNext, int episodes) {
        for (int ep = 0; ep < episodes; ep++) {
            List<Step> steps = trajectory(1000L + ep, 30);
            runEpisode(a, steps, greedyNext);
            runEpisode(b, steps, greedyNext);
        }
    }

    /** 두 두뇌의 Q-Table 두 벌과 epsilon이 <b>오차 0</b>으로 같은지 확인한다. */
    private static void assertSameTables(TabularBrain expected, TabularBrain actual, String what) {
        for (int c = 0; c < TabularBrain.COMMON_SIZE; c++) {
            assertArrayEquals(expected.qCommon[c], actual.qCommon[c], 0.0,
                    what + " — qCommon[" + c + "] 불일치");
        }
        for (int l = 0; l < TabularBrain.LOCAL_SIZE; l++) {
            assertArrayEquals(expected.qLocal[l], actual.qLocal[l], 0.0,
                    what + " — qLocal[" + l + "] 불일치");
        }
        assertEquals(expected.getEpsilon(), actual.getEpsilon(), 0.0, what + " — epsilon 불일치");
    }

    /** 학습이 실제로 일어났는지(테이블이 0에서 벗어났는지) — 둘 다 0이면 등가성 검사가 무의미해진다. */
    private static void assertLearned(TabularBrain brain) {
        for (double[] row : brain.qCommon) {
            for (double v : row) {
                if (v != 0.0) {
                    return;
                }
            }
        }
        throw new AssertionError("테이블이 전부 0 — 궤적이 학습에 반영되지 않았다(테스트가 헛돌고 있음)");
    }

    @Test
    @DisplayName("[DAY17] n-step SARSA(n=1) ≡ SARSA")
    void nStep_n1_은_SARSA와_같다() {
        NStepSARSA nStep = new NStepSARSA(42L, 7, 1);
        SARSA sarsa = new SARSA(42L, 7);

        runBoth(nStep, sarsa, false, 3);

        assertLearned(sarsa);
        assertSameTables(sarsa, nStep, "n=1");
    }

    @Test
    @DisplayName("[DAY17] n-step SARSA(n=∞) ≡ Monte Carlo")
    void nStep_n무한_은_MonteCarlo와_같다() {
        // 궤적(30스텝)보다 훨씬 큰 n → 창이 절대 안 차고 전부 꼬리(실제 return)로 마감된다 = MC
        NStepSARSA nStep = new NStepSARSA(42L, 7, 1000);
        MonteCarlo mc = new MonteCarlo(42L, 7);

        runBoth(nStep, mc, false, 3);

        assertLearned(mc);
        assertSameTables(mc, nStep, "n=∞");
    }

    @Test
    @DisplayName("[DAY20] Watkins Q(λ=0) ≡ Q-Learning")
    void qLambda_람다0_은_QLearning과_같다() {
        QLambda qLambda = new QLambda(42L, 7, 0.0);
        QLearning qLearning = new QLearning(42L, 7);

        runBoth(qLambda, qLearning, false, 3);

        assertLearned(qLearning);
        assertSameTables(qLearning, qLambda, "λ=0");
    }

    @Test
    @DisplayName("[DAY20] Watkins Q(λ) — 탐험이 없으면(항상 greedy) SARSA(λ)와 같다")
    void qLambda_항상greedy_는_SARSALambda와_같다() {
        // 다음 행동을 항상 greedy로 주면 Watkins cut이 한 번도 안 걸리고,
        // 부트스트랩도 max == 그 행동의 값이라 SARSA(λ)와 완전히 겹친다.
        QLambda qLambda = new QLambda(42L, 7, 0.9);
        SARSALambda sarsaLambda = new SARSALambda(42L, 7, 0.9);

        runBoth(qLambda, sarsaLambda, true, 3);

        assertLearned(sarsaLambda);
        assertSameTables(sarsaLambda, qLambda, "항상 greedy");
    }

    @Test
    @DisplayName("[DAY20] 반대로 탐험이 섞이면 Watkins cut 때문에 SARSA(λ)와 갈라진다")
    void qLambda_탐험이_섞이면_SARSALambda와_다르다() {
        // 위 테스트의 대조군 — cut이 "실제로 무언가를 하고 있음"을 확인한다.
        // (이게 없으면 cut을 통째로 지워도 위 테스트는 통과한다)
        QLambda qLambda = new QLambda(42L, 7, 0.9);
        SARSALambda sarsaLambda = new SARSALambda(42L, 7, 0.9);

        runBoth(qLambda, sarsaLambda, false, 3);

        boolean 갈라짐 = false;
        outer:
        for (int c = 0; c < TabularBrain.COMMON_SIZE; c++) {
            for (int a = 0; a < TabularBrain.ACTION_SIZE; a++) {
                if (qLambda.qCommon[c][a] != sarsaLambda.qCommon[c][a]) {
                    갈라짐 = true;
                    break outer;
                }
            }
        }
        assertEquals(true, 갈라짐, "탐험이 섞였는데도 SARSA(λ)와 같다면 Watkins cut이 작동하지 않는 것");
    }

    @Test
    @DisplayName("Monte Carlo는 부트스트랩을 쓰지 않는다(호출되면 예외)")
    void monteCarlo_bootstrap은_예외() {
        MonteCarlo mc = new MonteCarlo(42L, 7);
        org.junit.jupiter.api.Assertions.assertThrows(UnsupportedOperationException.class,
                () -> mc.bootstrap(0, 0, 0));
    }
}
