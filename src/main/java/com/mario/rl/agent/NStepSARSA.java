package com.mario.rl.agent;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

/**
 * n-step SARSA — <b>부트스트랩을 되돌린</b> on-policy TD. [DAY17 · 세로 심화 3막]
 *
 * <p><b>왜 n인가.</b> [DAY12] Monte Carlo는 부트스트랩을 <b>완전히 버리고</b> 에피소드가 끝나야 배웠다
 * (실제 return G로 일괄 갱신) → 신호 전파가 판 단위라 느리고, 1-1의 sparse-reward에서 TD에 밀렸다.
 * 반대로 1-step SARSA는 부트스트랩에 전적으로 기대 한 칸씩만 신용을 되돌린다. <b>n-step은 그 사이</b>다 —
 * 실제 보상을 <b>n칸까지만</b> 쌓고, 그 지점부터는 추정치로 잇는다.</p>
 *
 * <pre>
 * G = r_t + γ·r_{t+1} + … + γ^{n-1}·r_{t+n-1}  +  γ^n · Q(S_{t+n}, A_{t+n})
 *     └──────── 실제로 받은 보상 n개 ────────┘     └── 부트스트랩(SARSA와 동일) ──┘
 * </pre>
 *
 * <p><b>n이 곧 손잡이다</b>: {@code n=1}이면 정확히 {@link SARSA}, {@code n=∞}(에피소드 길이)면
 * {@link MonteCarlo}가 된다. 즉 이 클래스 하나가 MC와 TD를 잇는 다리이고, n을 줄이는 것이
 * "MC에 부트스트랩을 도로 넣는" 행위다.</p>
 *
 * <p><b>구현.</b> 최근 n스텝을 슬라이딩 창({@code window})에 담아두고, 창이 n개로 차면 <b>가장 오래된 스텝</b>
 * 하나를 위 식으로 갱신하며 창에서 뺀다(스텝마다 정확히 한 번 갱신, n스텝 지연). 에피소드가 끝나면 창에
 * 남은 꼬리(n칸이 안 되는 마지막 스텝들)는 부트스트랩할 미래가 없으므로 <b>MC처럼 실제 return으로</b> 마감한다.</p>
 *
 * <p>부트스트랩 규칙 자체는 SARSA와 <b>완전히 같다</b>(다음 실제 행동의 값). 오직 "몇 칸의 실제 보상을 쌓고
 * 부트스트랩하느냐"만 다르므로 {@link SARSA}를 상속해 {@link #bootstrap}은 그대로 쓰고
 * {@link #learn}·{@link #endEpisode}만 바꾼다([DAY15] {@link SARSALambda}와 같은 패턴).</p>
 */
public class NStepSARSA extends SARSA {

    /** 실제 보상을 몇 칸 쌓고 부트스트랩할지(n)의 기본값. [DAY17] SARSA(λ=0.9)의 유효 horizon(≈10스텝)과 대응. */
    private static final int DEFAULT_N = 8;

    /** 실제 보상을 쌓는 칸 수 (1이면 SARSA, 에피소드 길이면 Monte Carlo). */
    private final int n;
    /** 할인 거듭제곱 캐시: {@code gammaPow[i] = γ^i} (i = 0..n). */
    private final double[] gammaPow;

    /** 아직 갱신하지 못한 최근 스텝들(오래된 것이 head). 크기가 n이 되면 head를 갱신하며 뺀다. */
    private final Deque<Step> window = new ArrayDeque<>();

    /** 창에 담아두는 한 스텝 — (상태 두 인덱스, 행동, 그때 받은 보상). */
    private record Step(int common, int local, int action, double reward) {
    }

    /** 시드·행동 수·n 지정 생성자. */
    public NStepSARSA(long seed, int activeActions, int n) {
        super(seed, activeActions);
        this.n = requireValidN(n);
        this.gammaPow = discountPowers(this.n);
    }

    /** 시드·n 지정(전체 행동). 첫 인자가 {@code long}이면 <b>시드</b>다(행동 수 지정은 아래 {@code (int,int)} 생성자). */
    public NStepSARSA(long seed, int n) {
        super(seed);
        this.n = requireValidN(n);
        this.gammaPow = discountPowers(this.n);
    }

    /** 행동 수·n 지정(비결정적 시드). 첫 인자가 {@code int}면 <b>행동 수</b>다. */
    public NStepSARSA(int activeActions, int n) {
        super(activeActions);
        this.n = requireValidN(n);
        this.gammaPow = discountPowers(this.n);
    }

    /** n만 지정(비결정적 시드·전체 행동). */
    public NStepSARSA(int n) {
        super();
        this.n = requireValidN(n);
        this.gammaPow = discountPowers(this.n);
    }

    /** 기본값(n=8)으로 초기화. */
    public NStepSARSA() {
        this(DEFAULT_N);
    }

    /**
     * 한 스텝: 창에 쌓아두고, 창이 n칸으로 차면 <b>가장 오래된 스텝</b>을 n-step return으로 갱신한다.
     * 종료 스텝이면 창에 남은 꼬리를 실제 return으로 모두 마감한다(부트스트랩할 미래가 없음 = MC 꼬리).
     */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        window.addLast(new Step(common, local, action, reward));

        if (done) {
            flushTail();
            return;
        }
        if (window.size() == n) {
            updateOldest(nextCommon, nextLocal, nextAction);
        }
    }

    /** 에피소드 경계에서 창을 비운다(종료 스텝의 {@link #flushTail} 이후 안전망). 이후 ε 감쇠는 부모에 위임. */
    @Override
    public void endEpisode() {
        window.clear();
        super.endEpisode();
    }

    /**
     * 창의 가장 오래된 스텝 하나를 갱신하고 창에서 뺀다:
     * {@code G = Σ γ^i·r_i (i=0..n-1) + γ^n·bootstrap(S_{t+n}, A_{t+n})}.
     * (호출 시점에 창 크기는 정확히 n이고, {@code next*}가 곧 n스텝 뒤 상태·행동이다.)
     */
    private void updateOldest(int nextCommon, int nextLocal, int nextAction) {
        double g = 0.0;
        int i = 0;
        for (Step s : window) {  // ArrayDeque 순회 = 오래된 것부터
            g += gammaPow[i++] * s.reward();
        }
        g += gammaPow[n] * bootstrap(nextCommon, nextLocal, nextAction);

        Step oldest = window.removeFirst();
        updateToward(oldest.common(), oldest.local(), oldest.action(), g);
    }

    /**
     * 에피소드 종료: 창에 남은 스텝들(n칸을 못 채운 꼬리)을 <b>역방향 실제 return</b>으로 갱신한다.
     * 미래가 잘렸으니 부트스트랩 항이 사라져 [DAY12] Monte Carlo의 갱신과 정확히 같아진다.
     */
    private void flushTail() {
        double g = 0.0;
        Iterator<Step> backward = window.descendingIterator();
        while (backward.hasNext()) {
            Step s = backward.next();
            g = s.reward() + DISCOUNT_FACTOR * g;
            updateToward(s.common(), s.local(), s.action(), g);
        }
        window.clear();
    }

    /** {@code γ^0 … γ^n}을 미리 계산해 둔다(매 스텝 pow 호출 회피). */
    private static double[] discountPowers(int n) {
        double[] powers = new double[n + 1];
        powers[0] = 1.0;
        for (int i = 1; i <= n; i++) {
            powers[i] = powers[i - 1] * DISCOUNT_FACTOR;
        }
        return powers;
    }

    /** n은 1 이상이어야 한다(1 = 보통의 SARSA). */
    private static int requireValidN(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("n(step)은 1 이상이어야 함: " + n);
        }
        return n;
    }
}
