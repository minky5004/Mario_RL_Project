package com.mario.rl.agent;

import java.util.ArrayList;
import java.util.List;

/**
 * Watkins Q(λ) — 적격흔적(eligibility trace)을 얹은 <b>off-policy</b> TD. [DAY20 · 세로 심화 2막]
 *
 * <p><b>왜 QL에도 λ인가.</b> [DAY15]에서 SARSA에 흔적을 얹으니(SARSA(λ)) 분포째 완승했고,
 * [DAY17]에서 n-step도 이겼다 — 이 환경의 병목은 <b>신용 전파</b>다. 그런데 QL 계보는
 * [DAY16] Double-Q(값의 편향 축)가 후퇴한 뒤 아직 강화 성공이 없다. 같은 신용 전파 축을
 * QL에 적용한 교과서적 다음 버전이 Watkins Q(λ)다.</p>
 *
 * <p><b>SARSA(λ)와의 유일한 차이 — 흔적을 자른다(Watkins cut).</b> Q-Learning의 학습 목표는
 * "탐욕(greedy) 정책의 가치"다. 그런데 흔적은 <b>실제로 걸은 길</b>을 따라 신용을 되돌린다.
 * 에이전트가 탐험(비탐욕 행동)을 한 순간, 그 뒤의 경험은 더 이상 탐욕 정책의 경험이 아니므로
 * <b>그 앞의 흔적으로 전파하면 안 된다</b> — 다음 행동이 탐욕이면 흔적을 {@code γλ}로 감쇠해
 * 잇고, 탐험이면 전부 0으로 끊는다:</p>
 *
 * <pre>
 * δ = reward + γ·max_a( qCommon[nc][a] + qLocal[nl][a] ) − ( qCommon[c][a] + qLocal[l][a] )   // QL의 TD 오차
 * e(현재 s,a) ← 1                                        // replacing trace
 * 모든 (s,a):  Q(s,a) += α·δ·e(s,a)
 * 다음 행동 a'가 greedy(Q(s',a')=max)면  e ← γλ·e        // 흔적을 이어 전파
 * 아니면(탐험)                          e ← 0            // Watkins cut — 흔적 절단
 * 에피소드 끝: 모든 e ← 0
 * </pre>
 *
 * <p><b>양날의 검.</b> cut 덕분에 off-policy 목표가 오염되지 않지만, ε가 큰 초반엔 탐험이 잦아
 * 흔적이 몇 스텝 못 가 끊긴다 → λ의 이득이 초반에 제한될 수 있다(ε가 하한 0.01로 내려간 후반엔
 * 거의 안 끊겨 SARSA(λ)처럼 작동). 이게 이번 실험의 관전 포인트다.</p>
 *
 * <p>흔적 구조(두 테이블 {@code eCommon}·{@code eLocal}, replacing trace, 활성 목록으로 전체 스윕
 * 회피, {@code α/2·δ·e} 분할 갱신)는 {@link SARSALambda}와 동일하다. 부트스트랩은 {@link QLearning}
 * 그대로(max)이므로 상속해 {@link #learn}·{@link #endEpisode}만 바꾼다. λ 기본 0.9,
 * {@code -Dmario.lambda}로 조절(SARSA(λ)와 같은 손잡이).</p>
 */
public class QLambda extends QLearning {

    /** 적격흔적 감쇠 계수 λ 기본값(SARSA(λ)와 동일). [DAY20] */
    private static final double DEFAULT_LAMBDA = 0.9;
    /** 이 값 미만으로 감쇠한 흔적은 0으로 보고 활성 목록에서 뺀다(전 테이블 스윕 회피). */
    private static final double TRACE_MIN = 1e-4;

    /** 적격흔적 감쇠 계수 (γλ에서 λ). */
    private final double lambda;

    /** `Q_공통`·`Q_위치` 흔적 테이블(가치 테이블과 같은 모양). */
    private final double[][] eCommon = new double[COMMON_SIZE][ACTION_SIZE];
    private final double[][] eLocal = new double[LOCAL_SIZE][ACTION_SIZE];

    /** 흔적이 0이 아닌 (상태 인덱스, 행동) 쌍만 담는 활성 목록 — SARSA(λ)와 같은 스윕 회피. */
    private final List<int[]> activeCommon = new ArrayList<>();
    private final List<int[]> activeLocal = new ArrayList<>();

    /** 시드·행동 수·λ 지정 생성자. */
    public QLambda(long seed, int activeActions, double lambda) {
        super(seed, activeActions);
        this.lambda = lambda;
    }

    /** 시드·λ 지정(전체 행동). */
    public QLambda(long seed, double lambda) {
        super(seed);
        this.lambda = lambda;
    }

    /** 행동 수·λ 지정(비결정적 시드). */
    public QLambda(int activeActions, double lambda) {
        super(activeActions);
        this.lambda = lambda;
    }

    /** λ만 지정(비결정적 시드·전체 행동). */
    public QLambda(double lambda) {
        super();
        this.lambda = lambda;
    }

    /** 기본값(λ=0.9)으로 초기화. */
    public QLambda() {
        this(DEFAULT_LAMBDA);
    }

    /**
     * 한 스텝: QL TD 오차 δ(max 부트스트랩)를 구해, 현재 (s,a)에 흔적을 세우고(replacing)
     * 활성 흔적 전체에 {@code α/2·δ·e}를 적용한다. 그 뒤 <b>다음 행동이 greedy면 {@code γλ} 감쇠,
     * 탐험이면 흔적 절단(decay=0 → 전부 활성 목록에서 제거)</b> — Watkins cut.
     */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        double target = done
                ? reward
                : reward + DISCOUNT_FACTOR * bootstrap(nextCommon, nextLocal, nextAction);
        double delta = target - combined(common, local, action);

        // 현재 (s,a) 흔적 = 1 (replacing trace)
        setTrace(eCommon, activeCommon, common, action);
        setTrace(eLocal, activeLocal, local, action);

        // Watkins cut 판정: 다음 행동의 값이 max와 같으면 greedy(동점 포함), 아니면 탐험.
        // (done이면 흔적은 endEpisode에서 어차피 전부 지워지므로 decay 값은 무의미)
        boolean nextGreedy = done
                || combined(nextCommon, nextLocal, nextAction) == maxCombined(nextCommon, nextLocal);
        double decay = nextGreedy ? DISCOUNT_FACTOR * lambda : 0.0;

        propagate(qCommon, eCommon, activeCommon, delta, decay);
        propagate(qLocal, eLocal, activeLocal, delta, decay);
    }

    /** 에피소드 경계에서 흔적을 모두 지운다. 이후 ε 감쇠는 부모에 위임. */
    @Override
    public void endEpisode() {
        clearTraces(eCommon, activeCommon);
        clearTraces(eLocal, activeLocal);
        super.endEpisode();
    }

    /** replacing trace: 해당 셀 흔적을 1로 세운다. 0→비영이면 활성 목록에 편입. */
    private static void setTrace(double[][] e, List<int[]> active, int index, int action) {
        if (e[index][action] == 0.0) {
            active.add(new int[]{index, action});
        }
        e[index][action] = 1.0;
    }

    /** 활성 흔적마다 {@code q += α/2·δ·e} 적용 후 {@code e *= decay}. TRACE_MIN 미만이면 0으로 제거. */
    private static void propagate(double[][] q, double[][] e, List<int[]> active,
                                  double delta, double decay) {
        for (int i = active.size() - 1; i >= 0; i--) {
            int[] key = active.get(i);
            int index = key[0];
            int action = key[1];
            q[index][action] += TABLE_LR * delta * e[index][action];
            double decayed = e[index][action] * decay;
            if (decayed < TRACE_MIN) {
                e[index][action] = 0.0;
                active.remove(i);
            } else {
                e[index][action] = decayed;
            }
        }
    }

    /** 활성 목록에 남은 흔적 셀만 0으로 되돌리고 목록을 비운다. */
    private static void clearTraces(double[][] e, List<int[]> active) {
        for (int[] key : active) {
            e[key[0]][key[1]] = 0.0;
        }
        active.clear();
    }
}
