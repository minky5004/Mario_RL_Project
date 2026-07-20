package com.mario.rl;

import com.mario.rl.agent.Brain;
import com.mario.rl.agent.QLearning;
import com.mario.rl.agent.SARSA;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * {@code -Dmario.algo} → 두뇌 매핑({@link Main#createBrain}) 스모크 테스트.
 *
 * <p>알고리즘이 12종까지 늘면서 별칭({@code qlambda}·{@code double_q}·{@code genetic_v2} …)도 함께 늘었다.
 * 이름을 잘못 적으면 <b>조용히 {@code qlearning}으로 폴백</b>하므로(에러가 안 난다),
 * 5시드 학습 몇 시간을 돌리고 나서야 "어? 이거 QL이었네"를 알게 된다. 실제로 비교 챕터 전체가
 * 이 문자열 하나에 걸려 있으므로 전 이름을 표로 못박는다.</p>
 */
class MainBrainFactoryTest {

    /** 옵션 없이(시드·행동 수·하이퍼파라미터 미지정) 이름만으로 두뇌를 만든다. */
    private static Brain create(String algo) {
        return Main.createBrain(algo, null, null, null, null, null, null, null, false);
    }

    @ParameterizedTest(name = "-Dmario.algo={0} → {1}")
    @CsvSource({
            "qlearning,        QLearning",
            "q_lambda,         QLambda",
            "qlambda,          QLambda",
            "watkins_q_lambda, QLambda",
            "double_qlearning, DoubleQLearning",
            "double_q,         DoubleQLearning",
            "sarsa,            SARSA",
            "sarsa_lambda,     SARSALambda",
            "nstep_sarsa,      NStepSARSA",
            "n_step_sarsa,     NStepSARSA",
            "expected_sarsa,   ExpectedSARSA",
            "monte_carlo,      MonteCarlo",
            "random,           RandomBrain",
            "ga,               GeneticAlgorithm",
            "genetic,          GeneticAlgorithm",
            "ga_v2,            GeneticAlgorithmV2",
            "genetic_v2,       GeneticAlgorithmV2",
            "es,               EvolutionStrategy",
            "evolution_strategy, EvolutionStrategy",
    })
    @DisplayName("알고리즘 이름·별칭이 모두 올바른 두뇌를 만든다")
    void 이름_매핑(String algo, String expectedSimpleName) {
        assertEquals(expectedSimpleName, create(algo).getClass().getSimpleName(),
                "'" + algo + "' 매핑이 어긋났다 — 오타는 조용히 qlearning으로 폴백하므로 학습을 다 돌린 뒤에야 들킨다");
    }

    @Test
    @DisplayName("상속 관계도 의도대로다 (SARSA 계열 / QLearning 계열)")
    void 상속_관계() {
        // 세로 심화는 "기존 두뇌를 상속해 한 부분만 바꾼다"는 설계라 계열이 곧 의미다.
        assertInstanceOf(SARSA.class, create("sarsa_lambda"));   // [DAY15]
        assertInstanceOf(SARSA.class, create("nstep_sarsa"));    // [DAY17]
        assertInstanceOf(QLearning.class, create("q_lambda"));   // [DAY20]
    }

    @Test
    @DisplayName("알 수 없는 이름은 qlearning으로 폴백한다(기존 동작 유지)")
    void 알수없는_이름은_폴백() {
        assertInstanceOf(QLearning.class, create("존재하지않는알고리즘"));
        assertInstanceOf(QLearning.class, create(""));
    }

    @Test
    @DisplayName("시드를 주면 결정적이고, 다른 시드면 다른 궤적을 낸다")
    void 시드_라우팅() {
        // 5시드 병렬 비교([DAY9~])의 전제 — 같은 시드면 같은 실행이어야 한다.
        assertEquals(actionSequence(brainWithSeed(7L)), actionSequence(brainWithSeed(7L)));
        assertNotEquals(actionSequence(brainWithSeed(7L)), actionSequence(brainWithSeed(8L)));
    }

    @Test
    @DisplayName("-Dmario.actions=6 이면 긴 점프(6번)를 절대 고르지 않는다")
    void 행동_캡() {
        // [DAY9] 긴 점프 대조군용 손잡이. 캡이 새면 "6행동 baseline"이 몰래 7행동이 된다.
        Brain brain = Main.createBrain("qlearning", 42L, 6L, null, null, null, null, null, false);
        for (int i = 0; i < 2000; i++) {
            int action = brain.selectAction(i % 108, i % 1080);
            if (action < 0 || action >= 6) {
                throw new AssertionError("행동 캡(6)을 벗어난 행동: " + action);
            }
        }
    }

    /** 시드를 지정한 두뇌 하나. */
    private static Brain brainWithSeed(long seed) {
        return Main.createBrain("qlearning", seed, null, null, null, null, null, null, false);
    }

    /** epsilon=1.0(전부 무작위) 상태에서 뽑은 행동 나열 — 시드가 같으면 같아야 한다. */
    private static String actionSequence(Brain brain) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 50; i++) {
            sb.append(brain.selectAction(0, 0));
        }
        return sb.toString();
    }
}
