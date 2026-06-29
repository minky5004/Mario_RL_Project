package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.util.Random;

/**
 * 진화 전략(ES, OpenAI-ES 방식) — <b>모집단을 섞는 대신, 한 부모를 노이즈로 흔들어 기울기를 추정</b>하는 두뇌. [DAY14]
 *
 * <p>{@link GeneticAlgorithm GA}(DAY13)도 진화였지만 방식이 다르다. GA는 <b>여러 정책(모집단)을 토너먼트로
 * 골라 섞고(교배) 흔들어(돌연변이)</b> 세대를 넘긴다 — 여러 부모. ES는 <b>부모가 단 하나</b>다. 그 부모
 * {@code θ}에 가우시안 노이즈를 더한 자식 {@code θ+σ·ε}를 여럿 만들어 각각 한 판 굴리고, <b>잘한 노이즈
 * 방향으로 부모를 한 걸음 옮긴다</b>(적합도로 가중한 노이즈 평균 = 기울기 추정). 교배도 토너먼트도 없다.</p>
 *
 * <p>이 둘을 나란히 두는 이유: DAY13에서 GA가 바닥선(Random) <i>아래</i>로 떨어졌는데, 그게
 * <b>"진화 패밀리 자체의 한계"인지 "GA라는 특정 구현(교배·결정론 정책·8천 파라미터)의 탓"인지</b>를
 * 가르는 대조군이 ES다. 같은 정책 표현·같은 환경·같은 예산(1500판)에서 <b>진화의 다른 손잡이</b>를 쓴다.</p>
 *
 * <p><b>개체(자식) = 표 두뇌와 똑같은 두 테이블</b> {@code qCommon[108][7] + qLocal[1080][7]}의 실수
 * 가중치다(GA·TD/MC와 완전히 같은 정책 표현 → 학습 방식만 다른 공정 비교). 행동은 두 테이블 합산값의
 * argmax(탐험 ε 없음 — 탐험은 노이즈가 맡는다).</p>
 *
 * <p><b>적합도 = 그 판 보상 합</b>(GA와 동일). {@link #learn}이 reward를 누적하고, {@link #endEpisode}에서
 * 그 합을 현재 자식의 적합도로 확정한다. 전진보상 {@code +0.1×px} 덕에 사실상 도달 거리에 비례한다.</p>
 *
 * <p>{@link Brain}을 {@link RandomBrain}·{@link GeneticAlgorithm}처럼 <b>직접</b> 구현한다(부모·노이즈 여러
 * 벌이라, 한 벌만 드는 {@link TabularBrain}은 상속하지 않는다):</p>
 * <ul>
 *   <li>{@link #selectAction} = <b>지금 평가 중인 자식</b>({@code θ+σ·ε_i})의 정책으로 argmax.</li>
 *   <li>{@link #learn} = 스텝 학습 없음, 적합도(보상 합)만 누적.</li>
 *   <li>{@link #endEpisode} = 한 자식 평가 끝 → 적합도 기록 → 다음 자식. 한 세대를 다 굴리면 <b>부모 갱신</b>.</li>
 * </ul>
 *
 * <p>총 예산(1500)을 {@code 자식 수 × 세대}로 쪼갠다({@value #POPULATION_SIZE}×50=1500, GA와 동일).
 * {@link RLAgent}·소켓·프로토콜은 한 줄도 바뀌지 않는다.</p>
 */
public class EvolutionStrategy implements Brain {

    /** `Q_공통`(위치 무시) 상태 크기. */
    private static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** `Q_위치`(위치 포함) 상태 크기. */
    private static final int LOCAL_SIZE = StateEncoder.LOCAL_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 한 세대의 자식 수(λ) — 부모에서 노이즈로 파생되는 표본 수. 1500판 ÷ 30 = 50세대(GA와 동일 예산). */
    private static final int POPULATION_SIZE = 30;
    /**
     * 노이즈 세기(σ) — 부모에 더하는 가우시안 노이즈의 표준편차. 탐험 폭을 정한다.
     *
     * <p>처음 σ=0.1(GA <i>초기화</i> σ에 맞춤)로 돌렸더니 2/5 시드가 <b>제자리 collapse</b>했다 —
     * 부모가 출발 상태에서 argmax=NOOP(행동 0)로 수렴하면 σ=0.1로는 노이즈가 그걸 못 뒤집어
     * 모든 자식이 똑같이 timeout(−50) → 적합도 분산 0 → 기울기 0 → 탈출 불가능한 고정점.
     * ES의 섭동 σ는 GA의 <i>돌연변이</i> σ(0.5) 역할에 가까우므로 0.3으로 키워 NOOP 함정을 벗어나게 한다.</p>
     */
    private static final double SIGMA = 0.3;
    /** 학습률(α) — 추정한 기울기 방향으로 부모를 옮기는 한 걸음 크기. */
    private static final double LEARNING_RATE = 0.05;

    /** 난수 생성기(노이즈 생성 공용). */
    private final Random random;
    /** 실제로 쓰는 행동 수(≤ {@link #ACTION_SIZE}). 비교는 다른 두뇌와 동일하게 6. */
    private final int activeActions;

    /** 부모(θ) — 갱신 대상인 단 하나의 정책 가중치(두 테이블). */
    private final double[][] parentCommon = new double[COMMON_SIZE][ACTION_SIZE];
    private final double[][] parentLocal = new double[LOCAL_SIZE][ACTION_SIZE];

    /** 이번 세대 각 자식이 쓴 노이즈(ε) — 부모 갱신 때 적합도로 가중하려면 보관해야 한다. */
    private final double[][][] noiseCommon = new double[POPULATION_SIZE][COMMON_SIZE][ACTION_SIZE];
    private final double[][][] noiseLocal = new double[POPULATION_SIZE][LOCAL_SIZE][ACTION_SIZE];
    /** 이번 세대 각 자식의 적합도(F) = 그 판 보상 합. */
    private final double[] fitness = new double[POPULATION_SIZE];

    /** 지금 평가 중인 자식의 합성 정책(θ + σ·ε_i) — argmax는 이걸 본다(부모를 매 스텝 안 더하려고 캐시). */
    private final double[][] childCommon = new double[COMMON_SIZE][ACTION_SIZE];
    private final double[][] childLocal = new double[LOCAL_SIZE][ACTION_SIZE];

    /** 지금 평가 중인 자식 인덱스(0 ~ {@link #POPULATION_SIZE}-1). */
    private int currentIndex;
    /** 지금 평가 중인 자식이 이번 판에서 쌓은 보상 합(= 적합도 후보). */
    private double currentFitness;
    /** 현재 세대 번호(로그용, 1부터). */
    private int generation;

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public EvolutionStrategy() {
        this(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public EvolutionStrategy(long seed) {
        this(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public EvolutionStrategy(long seed, int activeActions) {
        this(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public EvolutionStrategy(int activeActions) {
        this(new Random(), activeActions);
    }

    private EvolutionStrategy(Random random, int activeActions) {
        if (activeActions < 1 || activeActions > ACTION_SIZE) {
            throw new IllegalArgumentException(
                    "activeActions 는 1~" + ACTION_SIZE + " 범위여야 함: " + activeActions);
        }
        this.random = random;
        this.activeActions = activeActions;
        // 부모는 0에서 출발(OpenAI-ES 표준) — 첫 세대 자식은 순수 노이즈가 정책을 정한다.
        this.currentIndex = 0;
        this.currentFitness = 0.0;
        this.generation = 1;
        prepareChild(0);
    }

    /** 지금 평가 중인 자식의 정책으로 행동을 고른다(두 테이블 합산 argmax, ε 없음). */
    @Override
    public int selectAction(int common, int local) {
        int bestIndex = 0;
        double bestValue = childCommon[common][0] + childLocal[local][0];
        for (int act = 1; act < activeActions; act++) {
            double v = childCommon[common][act] + childLocal[local][act];
            if (v > bestValue) {
                bestValue = v;
                bestIndex = act;
            }
        }
        return bestIndex;
    }

    /** ES는 스텝마다 학습하지 않는다 — 보상만 적합도로 누적한다(done은 {@link #endEpisode}에서 처리). */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        currentFitness += reward;
    }

    /**
     * 한 자식의 평가가 끝났다 — 적합도를 확정하고 다음 자식을 준비한다.
     * 한 세대(자식 {@value #POPULATION_SIZE}개)를 다 굴렸으면 부모를 갱신하고 새 세대로 넘어간다.
     */
    @Override
    public void endEpisode() {
        fitness[currentIndex] = currentFitness;
        currentFitness = 0.0;

        currentIndex++;
        if (currentIndex >= POPULATION_SIZE) {
            updateParent();
            currentIndex = 0;
            generation++;
        }
        prepareChild(currentIndex);
    }

    /**
     * 부모를 한 걸음 옮긴다(OpenAI-ES): 적합도로 가중한 노이즈 평균 = 기울기 추정.
     *
     * <p>{@code θ ← θ + α/(λσ) · Σ F̂_i · ε_i}. 적합도 {@code F}를 평균0·표준편차1로 정규화(F̂)해
     * 스케일·시드 간 보상 크기 차이에 둔감하게 만든 뒤, "평균보다 잘한 자식의 노이즈 방향"으로 부모를 민다.</p>
     */
    private void updateParent() {
        // 세대 통계 로그(일지/그래프용) + 정규화에 쓸 평균.
        double sum = 0.0;
        double best = fitness[0];
        for (double f : fitness) {
            sum += f;
            if (f > best) {
                best = f;
            }
        }
        double mean = sum / POPULATION_SIZE;
        System.out.printf("[ES] gen=%d best=%.1f mean=%.1f%n", generation, best, mean);

        // 적합도 정규화(평균0·표준편차1). 표준편차가 0이면(모든 자식 동점) 기울기 신호가 없어 갱신을 건너뛴다.
        double var = 0.0;
        for (double f : fitness) {
            double d = f - mean;
            var += d * d;
        }
        double std = Math.sqrt(var / POPULATION_SIZE);
        if (std < 1e-8) {
            return;
        }

        // θ ← θ + α/(λσ) · Σ F̂_i · ε_i  (가중치 F̂_i = (F_i-mean)/std 를 노이즈에 곱해 누적).
        double step = LEARNING_RATE / (POPULATION_SIZE * SIGMA);
        for (int i = 0; i < POPULATION_SIZE; i++) {
            double w = step * (fitness[i] - mean) / std;
            accumulate(parentCommon, noiseCommon[i], w);
            accumulate(parentLocal, noiseLocal[i], w);
        }
    }

    /** {@code table += w · noise} (부모 갱신 한 항). */
    private void accumulate(double[][] table, double[][] noise, double w) {
        for (int s = 0; s < table.length; s++) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                table[s][act] += w * noise[s][act];
            }
        }
    }

    /**
     * {@code i}번째 자식을 준비한다 — 노이즈 {@code ε_i}를 새로 뽑아 보관하고, 합성 정책 {@code θ+σ·ε_i}를 캐시한다.
     * 이후 {@link #selectAction}은 이 캐시({@link #childCommon}/{@link #childLocal})만 본다.
     */
    private void prepareChild(int i) {
        sampleAndCompose(parentCommon, noiseCommon[i], childCommon);
        sampleAndCompose(parentLocal, noiseLocal[i], childLocal);
    }

    /** 각 칸에 가우시안 노이즈를 뽑아 {@code noise}에 저장하고 {@code child = parent + σ·noise}를 채운다. */
    private void sampleAndCompose(double[][] parent, double[][] noise, double[][] child) {
        for (int s = 0; s < parent.length; s++) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                double eps = random.nextGaussian();
                noise[s][act] = eps;
                child[s][act] = parent[s][act] + SIGMA * eps;
            }
        }
    }

    /** ES는 ε 탐험이 없다 — 탐험은 노이즈가 맡는다. 로그·그래프엔 0.0으로 보고한다. */
    @Override
    public double getEpsilon() {
        return 0.0;
    }
}
