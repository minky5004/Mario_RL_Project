package com.mario.rl.agent;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * SARSA(λ) — 적격흔적(eligibility trace)을 얹은 on-policy TD. [DAY15 · 세로 심화 2막]
 *
 * <p><b>왜 λ인가.</b> 보통의 SARSA(1-step TD)는 한 스텝의 TD 오차를 <b>바로 직전 상태</b>에만 반영한다.
 * 1-1처럼 보상이 뒤에 한 번(깃발 +1000) 몰린 <b>sparse-reward</b> 환경에선, 그 신호가 출발선까지
 * 스며들려면 여러 에피소드에 걸쳐 한 칸씩 역전파돼야 한다(느림). 적격흔적은 <b>최근에 밟은 상태들에
 * "책임 꼬리표"(trace)를 남겨</b>, 한 번의 TD 오차를 그 꼬리표만큼 <b>과거로 한꺼번에 되돌려 전파</b>한다
 * → 보상 신용 할당이 빨라진다. Monte Carlo(λ=1, 부트스트랩 없음)와 1-step SARSA(λ=0) 사이를 λ로 잇는다.</p>
 *
 * <pre>
 * δ = reward + γ·( qCommon[nc][na] + qLocal[nl][na] ) − ( qCommon[c][a] + qLocal[l][a] )   // SARSA의 TD 오차
 * e(현재 s,a) ← 1                                                                          // replacing trace
 * 모든 (s,a):  Q(s,a) += α·δ·e(s,a) ;  e(s,a) ← γλ·e(s,a)                                  // 꼬리표만큼 과거로 전파 + 감쇠
 * 에피소드 끝: 모든 e ← 0
 * </pre>
 *
 * <p><b>두 테이블에서의 흔적.</b> 이 프로젝트는 가치가 두 테이블 합 {@code Q=qCommon+qLocal}이므로,
 * 흔적도 테이블별로 {@code eCommon}·{@code eLocal}을 따로 둔다. 어떤 스텝에서 상태가 (c,l)·행동 a면
 * {@code eCommon[c][a]}·{@code eLocal[l][a]}에 동시에 꼬리표를 남기고, TD 오차 δ는 각 테이블의
 * 흔적에 비례해 {@code α/2·δ·e}로 갱신한다({@link TabularBrain#TABLE_LR} = α/2, 두 테이블 분할과 일관).</p>
 *
 * <p><b>replacing trace</b>(재방문 시 흔적을 누적 +1이 아니라 1로 재설정)를 쓴다 — 반복 방문에 흔적이 폭주하지
 * 않아 sparse-reward에서 더 안정적이다(Sutton 기본). λ 기본값 0.9(강한 전파), {@code -Dmario.lambda}로 조절.</p>
 *
 * <p>SARSA와 <b>부트스트랩은 완전히 같다</b>(다음 실제 행동의 값). 오직 "갱신을 흔적만큼 과거로 퍼뜨리는가"만
 * 다르므로 {@link SARSA}를 상속해 {@link #bootstrap}은 그대로 쓰고 {@link #learn}·{@link #endEpisode}만 바꾼다.</p>
 */
public class SARSALambda extends SARSA {

    /** 적격흔적 감쇠 계수 λ 기본값(1-step SARSA=0 ~ Monte Carlo=1 사이). [DAY15] */
    private static final double DEFAULT_LAMBDA = 0.9;
    /** 이 값 미만으로 감쇠한 흔적은 0으로 보고 활성 목록에서 뺀다(전 테이블 스윕 회피). */
    private static final double TRACE_MIN = 1e-4;

    /** 적격흔적 감쇠 계수 (γλ에서 λ). */
    private final double lambda;

    /** `Q_공통`·`Q_위치` 흔적 테이블(가치 테이블과 같은 모양). */
    private final double[][] eCommon = new double[COMMON_SIZE][ACTION_SIZE];
    private final double[][] eLocal = new double[LOCAL_SIZE][ACTION_SIZE];

    /**
     * 흔적이 0이 아닌 (상태 인덱스, 행동) 쌍만 담는 활성 목록 — 매 스텝 이 목록만 갱신·감쇠해
     * 전체 테이블(약 7천 칸) 스윕을 피한다. {@code γλ}로 감쇠하다 {@link #TRACE_MIN} 아래면 뺀다.
     */
    private final List<int[]> activeCommon = new ArrayList<>();
    private final List<int[]> activeLocal = new ArrayList<>();

    /** 시드·행동 수·λ 지정 생성자. */
    public SARSALambda(long seed, int activeActions, double lambda) {
        super(seed, activeActions);
        this.lambda = lambda;
    }

    /** 시드·λ 지정(전체 행동). */
    public SARSALambda(long seed, double lambda) {
        super(seed);
        this.lambda = lambda;
    }

    /** 행동 수·λ 지정(비결정적 시드). */
    public SARSALambda(int activeActions, double lambda) {
        super(activeActions);
        this.lambda = lambda;
    }

    /** λ만 지정(비결정적 시드·전체 행동). */
    public SARSALambda(double lambda) {
        super();
        this.lambda = lambda;
    }

    /** 기본값(λ=0.9)으로 초기화. */
    public SARSALambda() {
        this(DEFAULT_LAMBDA);
    }

    /**
     * 한 스텝: SARSA TD 오차 δ를 구해, 현재 (s,a)에 흔적을 세우고(replacing) 활성 흔적 전체에
     * {@code α/2·δ·e}를 적용한 뒤 {@code γλ}로 감쇠한다.
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

        // 활성 흔적 전체에 δ 전파 후 γλ 감쇠
        double decay = DISCOUNT_FACTOR * lambda;
        propagate(qCommon, eCommon, activeCommon, delta, decay);
        propagate(qLocal, eLocal, activeLocal, delta, decay);
    }

    /** 에피소드 경계에서 흔적을 모두 지운다(스텝 간 책임은 판을 넘기지 않음). 이후 ε 감쇠는 부모에 위임. */
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
