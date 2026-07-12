package com.mario.rl.agent;

import java.util.Random;

/**
 * Double Q-Learning — <b>최대화 편향(overestimation)을 없앤</b> off-policy TD. [DAY16 · 세로 심화 2막]
 *
 * <p><b>왜 Double인가.</b> 보통 Q-Learning은 다음 상태를 {@code max_a Q(s',a)}로 부트스트랩한다.
 * 그런데 Q값은 잡음이 섞인 <b>추정치</b>라, "최댓값을 고르는" 행위 자체가 우연히 크게 나온 추정을
 * 계속 집어 든다 → 값이 실제보다 <b>부풀려진다</b>(maximization bias). Double Q-Learning은
 * 가치 테이블을 <b>두 벌(Q_A, Q_B)</b>로 나눠, "어느 행동이 최선인지 고르는 테이블"과
 * "그 행동의 값을 읽는 테이블"을 <b>분리</b>한다 → 같은 잡음으로 뽑고 같은 잡음으로 재는 자기참조가 끊겨
 * 편향이 사라진다.</p>
 *
 * <pre>
 * 매 스텝 동전 던지기(50/50):
 *   A 갱신:  a* = argmax_a Q_A(s',a)                        // 고르기는 A로
 *            target = reward + γ · Q_B(s', a*)              // 값 읽기는 B로 (done이면 target=reward)
 *            δ = target − Q_A(s,a) ;  Q_A(s,a) += α·δ
 *   B 갱신:  대칭(고르기 B, 값 읽기 A)
 * </pre>
 *
 * <p><b>두 테이블 구조 위에서의 Double.</b> 이 프로젝트의 가치는 이미 두 테이블 합
 * {@code Q = qCommon + qLocal}(타일/coarse coding)이다. 그래서 "Double"은 이 합 단위로 얹혀,
 * {@code Q_A = qCommonA + qLocalA} · {@code Q_B = qCommonB + qLocalB}로 <b>총 4벌</b>이 된다.
 * A 쪽은 {@link TabularBrain}의 기존 {@code qCommon}·{@code qLocal}을 그대로 재사용하고, B 쌍만 여기서 더한다.
 * 각 벌 안의 α/2 분할 갱신은 그대로 유지({@link TabularBrain#TABLE_LR}).</p>
 *
 * <p><b>행동 선택.</b> ε-greedy의 greedy는 두 추정을 합친 {@code Q_A + Q_B}로 고른다(Sutton 표준의 합/평균).
 * 갱신 규칙이 QL과 근본적으로 달라(부트스트랩을 "고르기/읽기 분리"로 바꿈) {@link #bootstrap}을 쓰지 않고
 * {@link #learn}을 직접 오버라이드한다({@link MonteCarlo}처럼 bootstrap은 호출되면 버그 → 예외).</p>
 */
public class DoubleQLearning extends TabularBrain {

    /** 두 번째 가치 테이블 Q_B(첫 번째 A는 부모의 {@code qCommon}·{@code qLocal} 재사용). */
    private final double[][] qCommonB = new double[COMMON_SIZE][ACTION_SIZE];
    private final double[][] qLocalB = new double[LOCAL_SIZE][ACTION_SIZE];

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public DoubleQLearning() {
        super(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public DoubleQLearning(long seed) {
        super(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public DoubleQLearning(long seed, int activeActions) {
        super(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public DoubleQLearning(int activeActions) {
        super(new Random(), activeActions);
    }

    /** greedy는 두 추정의 합 {@code Q_A + Q_B}로 고른다(그 외 탐험은 부모와 동일). */
    @Override
    public int selectAction(int common, int local) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(activeActions);
        }
        return argMaxSum(common, local);
    }

    /**
     * 한 스텝: 동전을 던져 두 테이블 중 하나만 갱신한다.
     * "최선 행동 고르기"와 "그 값 읽기"를 서로 다른 테이블에서 하여 최대화 편향을 없앤다.
     */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        if (random.nextBoolean()) {
            // A 갱신: a* 는 A로 고르고 값은 B로 읽는다.
            updatePair(qCommon, qLocal, qCommonB, qLocalB,
                    common, local, action, reward, nextCommon, nextLocal, done);
        } else {
            // B 갱신: 대칭(a* 는 B로 고르고 값은 A로 읽는다).
            updatePair(qCommonB, qLocalB, qCommon, qLocal,
                    common, local, action, reward, nextCommon, nextLocal, done);
        }
    }

    /**
     * {@code updated} 테이블 쌍을 한 걸음 갱신한다 — 다음 상태 최선 행동은 {@code updated}로 고르고,
     * 그 행동의 값은 {@code evaluated} 테이블 쌍에서 읽어 부트스트랩한다.
     */
    private void updatePair(double[][] updCommon, double[][] updLocal,
                            double[][] evalCommon, double[][] evalLocal,
                            int common, int local, int action, double reward,
                            int nextCommon, int nextLocal, boolean done) {
        double target = reward;
        if (!done) {
            int aStar = argMax(updCommon, updLocal, nextCommon, nextLocal);
            target += DISCOUNT_FACTOR * (evalCommon[nextCommon][aStar] + evalLocal[nextLocal][aStar]);
        }
        double delta = target - (updCommon[common][action] + updLocal[local][action]);
        updCommon[common][action] += TABLE_LR * delta;
        updLocal[local][action] += TABLE_LR * delta;
    }

    /** 한 테이블 쌍 합 {@code common[c][a]+local[l][a]}이 최대인 행동(쓰는 행동 범위 내, 동점은 앞 인덱스). */
    private int argMax(double[][] common, double[][] local, int c, int l) {
        int best = 0;
        double bestValue = common[c][0] + local[l][0];
        for (int a = 1; a < activeActions; a++) {
            double v = common[c][a] + local[l][a];
            if (v > bestValue) {
                bestValue = v;
                best = a;
            }
        }
        return best;
    }

    /** 두 테이블 합 {@code Q_A + Q_B = (qCommon+qCommonB)+(qLocal+qLocalB)}이 최대인 행동. */
    private int argMaxSum(int common, int local) {
        int best = 0;
        double bestValue = sumAll(common, local, 0);
        for (int a = 1; a < activeActions; a++) {
            double v = sumAll(common, local, a);
            if (v > bestValue) {
                bestValue = v;
                best = a;
            }
        }
        return best;
    }

    /** {@code Q_A(s,a) + Q_B(s,a)} = 네 테이블 합. */
    private double sumAll(int common, int local, int action) {
        return qCommon[common][action] + qLocalB[local][action]
                + qCommonB[common][action] + qLocal[local][action];
    }

    /** Double Q-Learning은 단일 부트스트랩을 쓰지 않는다(고르기/읽기를 두 테이블로 분리) — 호출되면 학습 루프 버그. */
    @Override
    protected double bootstrap(int nextCommon, int nextLocal, int nextAction) {
        throw new UnsupportedOperationException(
                "Double Q-Learning은 단일 bootstrap을 쓰지 않는다 — learn이 두 테이블(고르기/읽기 분리)로 직접 갱신한다.");
    }
}
