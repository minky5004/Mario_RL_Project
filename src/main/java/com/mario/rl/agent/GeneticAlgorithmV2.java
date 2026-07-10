package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.util.Random;

/**
 * 유전 알고리즘 v2 — <b>DAY13 GA의 실패 원인 셋을 한꺼번에 고친 개편판.</b> [DAY18]
 *
 * <p>[DAY13] GA는 <b>무학습 바닥선(Random)보다도 못한 유일한 학습기</b>였다(깃발 0 · 후반거리 603 &lt; Random 654 ·
 * timeout 1122 = 챕터 최고). 일지의 진단은 세 가지의 <i>합작</i>이었고, v2는 그 셋을 각각 정면으로 고친다:</p>
 *
 * <table border="1">
 *   <caption>DAY13 진단 → v2 처방</caption>
 *   <tr><th>진단(DAY13)</th><th>처방(v2)</th></tr>
 *   <tr>
 *     <td>ⓐ <b>결정론 정책</b>(ε 없는 argmax) — 한 번 막히면 그 판 내내 같은 행동을 반복해 못 빠져나옴.
 *         매 스텝 주사위를 굴리는 Random이 오히려 탈출했다.</td>
 *     <td><b>softmax 샘플링</b>({@link #TEMPERATURE}) — 행동을 확률로 고른다. 막혀도 다른 행동이 나올 여지가 생긴다.
 *         ε처럼 탐험률을 고정하지 않고 <i>가중치 크기 자체가 확신도</i>가 되므로, 진화가
 *         "얼마나 결정론적으로 굴지"까지 스스로 정한다(가중치가 커질수록 argmax에 수렴).</td>
 *   </tr>
 *   <tr>
 *     <td>ⓑ <b>파라미터 8,316개</b>({@code 108×7 + 1080×7})를 30개체×50세대=1500 평가로는 못 훑음
 *         (파라미터당 평가 0.18회).</td>
 *     <td><b>{@code Q_위치}(7,560개) 제거 → {@code Q_공통}(756개)만 진화</b> = 11배 축소
 *         (파라미터당 평가 2.0회). 진화는 그래디언트가 없어 샘플 효율이 파라미터 수에 민감하다.
 *         위치 무시 상태(벽·구덩이·굼바 거리 + 수직 방향)만으로도 정책은 표현된다
 *         ([DAY7 트라이1]이 36칸만으로 깃발을 세 번 들었다).</td>
 *   </tr>
 *   <tr>
 *     <td>ⓒ <b>적합도 = 보상 합</b>인데 timeout(−50)이 사망(−100)보다 덜 아파서, 진화가
 *         "안 죽고 버티되 멀리 안 가는 겁쟁이"를 골랐음.</td>
 *     <td><b>적합도 = 그 판 도달 거리(maxX)</b>({@link #observeEpisodeOutcome}). 버티기는 점수를 못 준다 —
 *         오직 <i>멀리 간 개체</i>만 살아남는다. 깃발은 maxX 최대라 자동으로 최고점이 된다.</td>
 *   </tr>
 * </table>
 *
 * <p><b>바꾸지 않은 것</b>(DAY13과 동일 — 비교의 축을 흐리지 않기 위해): 예산 1500판, 엘리트 {@value #ELITE_COUNT},
 * 토너먼트 {@value #TOURNAMENT_SIZE}, 균등 교배, 돌연변이(유전자별 {@value #MUTATION_RATE} 확률 ·
 * σ={@value #MUTATION_SIGMA}), 초기화 σ={@value #INIT_SIGMA}.</p>
 *
 * <p><b>[DAY18 트라이2] 적합도 노이즈 제거 — {@code evalsPerGenome}.</b> 트라이1(개체당 1판)에서
 * <b>깃발을 든 엘리트가 다음 세대에 재현되지 않는</b> 현상이 나왔다(gen19 best=3161 → gen20 best=1425).
 * 처방ⓐ(softmax)가 확률 정책이라 <b>같은 genome도 매 판 다른 궤적</b>을 내는데 적합도는 <b>단 한 판의 표본</b>이라,
 * 엘리트·토너먼트가 <i>실력</i>이 아니라 <b>그 판의 운</b>을 뽑고 있었던 것. 트라이2는 같은 개체를
 * {@code evalsPerGenome}판 굴려 <b>평균</b>을 적합도로 삼는다(노이즈 1/√n). 단 <b>예산이 고정</b>이라
 * 공짜가 아니다 — {@code 모집단 × 세대 × 평가판수 = 1500}이므로 평가를 3판으로 늘리면 <b>세대가 절반</b>이 된다
 * (트라이1: 30×50×1 / 트라이2: 20×25×3). <b>적합도 정확도 ↔ 진화 스텝 수</b>의 교환이 이 트라이의 물음이다.</p>
 *
 * <p>{@link Brain}을 직접 구현한다({@link GeneticAlgorithm}·{@link RandomBrain}과 같은 이유 —
 * 모집단은 테이블 여러 벌이라 한 벌만 드는 {@link TabularBrain}과 맞지 않는다).</p>
 */
public class GeneticAlgorithmV2 implements Brain {

    /** `Q_공통`(위치 무시) 상태 크기 — v2가 진화시키는 <b>유일한</b> 테이블. */
    private static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 기본 모집단 크기 — 한 세대의 개체 수. 1500판 ÷ (30 × 1판) = 50세대. [DAY13과 동일 = 트라이1] */
    private static final int DEFAULT_POPULATION_SIZE = 30;
    /** 기본 평가 판수 — 개체 하나를 몇 판 굴려 적합도를 매기나. 1 = 한 판(트라이1). [DAY18 트라이2에서 3으로] */
    private static final int DEFAULT_EVALS_PER_GENOME = 1;
    /** 엘리트 수 — 적합도 상위 이만큼은 다음 세대로 그대로 살린다. [DAY13과 동일] */
    private static final int ELITE_COUNT = 2;
    /** 토너먼트 선택 크기. [DAY13과 동일] */
    private static final int TOURNAMENT_SIZE = 3;
    /** 돌연변이율 — 자식의 각 유전자가 흔들릴 확률. [DAY13과 동일] */
    private static final double MUTATION_RATE = 0.05;
    /** 돌연변이 세기 — 흔들 때 더하는 가우시안 노이즈의 표준편차. [DAY13과 동일] */
    private static final double MUTATION_SIGMA = 0.5;
    /** 초기화 노이즈 세기. [DAY13과 동일] */
    private static final double INIT_SIGMA = 0.1;

    /**
     * softmax 온도 τ — 1.0 고정(하이퍼파라미터를 늘리지 않는다). [DAY18 처방ⓐ]
     *
     * <p>τ를 1로 두면 <b>가중치 스케일이 곧 확신도</b>가 된다: 초기(σ=0.1)엔 로짓 차가 작아 거의 균등 무작위로
     * 시작하고(= 막혀도 빠져나올 수 있음), 진화가 잘하는 개체의 가중치를 키우면 저절로 argmax에 가까워진다.
     * "탐험을 얼마나 할지"를 사람이 정하지 않고 진화에 맡기는 셈.</p>
     */
    private static final double TEMPERATURE = 1.0;

    /** 난수 생성기(초기화·선택·교배·돌연변이·행동 샘플링 공용). */
    private final Random random;
    /** 실제로 쓰는 행동 수(≤ {@link #ACTION_SIZE}). 비교는 다른 두뇌와 동일하게 6. */
    private final int activeActions;
    /** 모집단 크기(한 세대의 개체 수). [DAY18 트라이2] 평가 판수를 늘리면 예산상 이걸 줄여야 한다. */
    private final int populationSize;
    /** 개체 하나를 몇 판 굴려 적합도를 매기나. 1=한 판(트라이1) · 3=세 판 평균(트라이2). [DAY18 트라이2] */
    private final int evalsPerGenome;

    /** 현재 세대의 모집단(개체 = genome). */
    private Genome[] population;
    /** 지금 평가 중인 개체의 인덱스. */
    private int currentIndex;
    /** 이번 판에 도달한 최대 x — {@link #observeEpisodeOutcome}가 채우고 {@link #endEpisode}가 적합도에 반영한다. */
    private int currentMaxX;
    /** 현재 개체를 지금까지 몇 판 평가했나(0 ~ {@link #evalsPerGenome}-1). [DAY18 트라이2] */
    private int evalCount;
    /** 현재 개체의 평가 판들에서 나온 maxX 합(평균 내려고 누적). [DAY18 트라이2] */
    private double fitnessSum;
    /** 현재 세대 번호(로그용, 1부터). */
    private int generation;
    /** 지금까지 본 최고 적합도(로그용). */
    private double bestFitnessSoFar;

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public GeneticAlgorithmV2() {
        this(new Random(), ACTION_SIZE, DEFAULT_POPULATION_SIZE, DEFAULT_EVALS_PER_GENOME);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public GeneticAlgorithmV2(long seed) {
        this(new Random(seed), ACTION_SIZE, DEFAULT_POPULATION_SIZE, DEFAULT_EVALS_PER_GENOME);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public GeneticAlgorithmV2(long seed, int activeActions) {
        this(new Random(seed), activeActions, DEFAULT_POPULATION_SIZE, DEFAULT_EVALS_PER_GENOME);
    }

    /** 시드 없이 행동 수만 지정. */
    public GeneticAlgorithmV2(int activeActions) {
        this(new Random(), activeActions, DEFAULT_POPULATION_SIZE, DEFAULT_EVALS_PER_GENOME);
    }

    /**
     * 모집단·평가 판수까지 지정. [DAY18 트라이2]
     *
     * <p>총 예산(1500판)이 고정이므로 {@code 모집단 × 세대 × 평가판수 = 1500}이 되도록 함께 정해야 한다 —
     * 평가 판수를 늘려 적합도 노이즈를 줄이는 대가는 <b>세대 수 감소</b>다(트라이2: 20 × 25세대 × 3판).</p>
     */
    public GeneticAlgorithmV2(long seed, int activeActions, int populationSize, int evalsPerGenome) {
        this(new Random(seed), activeActions, populationSize, evalsPerGenome);
    }

    private GeneticAlgorithmV2(Random random, int activeActions, int populationSize, int evalsPerGenome) {
        if (activeActions < 1 || activeActions > ACTION_SIZE) {
            throw new IllegalArgumentException(
                    "activeActions 는 1~" + ACTION_SIZE + " 범위여야 함: " + activeActions);
        }
        if (populationSize <= ELITE_COUNT) {
            throw new IllegalArgumentException(
                    "populationSize 는 엘리트 수(" + ELITE_COUNT + ")보다 커야 함: " + populationSize);
        }
        if (evalsPerGenome < 1) {
            throw new IllegalArgumentException("evalsPerGenome 는 1 이상이어야 함: " + evalsPerGenome);
        }
        this.random = random;
        this.activeActions = activeActions;
        this.populationSize = populationSize;
        this.evalsPerGenome = evalsPerGenome;
        this.population = new Genome[populationSize];
        for (int i = 0; i < populationSize; i++) {
            population[i] = randomGenome();
        }
        this.currentIndex = 0;
        this.currentMaxX = 0;
        this.evalCount = 0;
        this.fitnessSum = 0.0;
        this.generation = 1;
        this.bestFitnessSoFar = Double.NEGATIVE_INFINITY;
    }

    /**
     * 지금 평가 중인 개체의 정책으로 행동을 <b>확률적으로</b> 고른다. [처방ⓐ]
     *
     * <p>{@code Q_공통} 한 테이블의 값을 로짓으로 보고 softmax 확률을 만들어 샘플링한다
     * ({@code local}은 v2가 쓰지 않는다 — 위치 테이블을 없앴으므로). DAY13의 argmax와 달리,
     * 같은 상태에서도 매번 같은 행동이 나오지 않아 "막혀서 그 판 내내 timeout"이 구조적으로 사라진다.</p>
     */
    @Override
    public int selectAction(int common, int local) {
        double[] logits = population[currentIndex].qCommon[common];

        // 오버플로 방지 — 최댓값을 빼고 exp (softmax는 상수 이동에 불변).
        double max = logits[0];
        for (int a = 1; a < activeActions; a++) {
            if (logits[a] > max) {
                max = logits[a];
            }
        }

        double sum = 0.0;
        double[] probs = new double[activeActions];
        for (int a = 0; a < activeActions; a++) {
            probs[a] = Math.exp((logits[a] - max) / TEMPERATURE);
            sum += probs[a];
        }

        // 누적 확률로 룰렛 샘플링.
        double dice = random.nextDouble() * sum;
        double acc = 0.0;
        for (int a = 0; a < activeActions; a++) {
            acc += probs[a];
            if (dice < acc) {
                return a;
            }
        }
        return activeActions - 1;  // 부동소수 오차 방어.
    }

    /** GA는 스텝마다 학습하지 않는다. v2는 <b>보상도 보지 않는다</b> — 적합도가 maxX이기 때문. [처방ⓒ] */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        // 의도적으로 비어 있음.
    }

    /** 이번 판의 도달 거리를 받아둔다 — 이게 곧 현재 개체의 적합도다. [처방ⓒ] */
    @Override
    public void observeEpisodeOutcome(int maxX) {
        this.currentMaxX = maxX;
    }

    /**
     * 한 판이 끝났다 — 도달 거리를 현재 개체의 적합도에 누적한다.
     *
     * <p>[DAY18 트라이1]은 {@code evalsPerGenome=1}이라 한 판이 곧 적합도였다. <b>트라이2는 같은 개체를
     * {@code evalsPerGenome}판 굴려 <u>평균</u>을 적합도로 삼는다</b> — softmax가 확률 정책이라 같은 genome도
     * 매 판 다른 궤적을 내므로, 한 판 표본은 "실력"이 아니라 "그 판의 운"을 재기 때문이다(트라이1에서
     * 깃발을 든 엘리트가 다음 세대에 재현되지 않았다). 평균을 내면 노이즈가 1/√n으로 줄어
     * 엘리트·토너먼트가 <b>운이 아니라 실력</b>을 뽑는다.</p>
     *
     * <p>개체를 다 평가했으면 다음 개체로, 모집단을 다 돌았으면 세대 진화를 수행한다.</p>
     */
    @Override
    public void endEpisode() {
        fitnessSum += currentMaxX;
        currentMaxX = 0;
        evalCount++;

        if (evalCount < evalsPerGenome) {
            return;  // 같은 개체를 한 판 더 굴린다.
        }

        double fitness = fitnessSum / evalsPerGenome;
        population[currentIndex].fitness = fitness;
        if (fitness > bestFitnessSoFar) {
            bestFitnessSoFar = fitness;
        }
        fitnessSum = 0.0;
        evalCount = 0;

        currentIndex++;
        if (currentIndex >= populationSize) {
            evolve();
            currentIndex = 0;
            generation++;
        }
    }

    /**
     * 한 세대를 진화시킨다 — 엘리트 보존 + 토너먼트 선택 + 균등 교배 + 돌연변이.
     * (연산자는 [DAY13]과 동일. 바뀐 건 <i>무엇을 적합도로 보느냐</i>와 <i>테이블 크기</i>뿐이다.)
     */
    private void evolve() {
        int bestIdx = 0;
        double sum = 0.0;
        for (int i = 0; i < populationSize; i++) {
            sum += population[i].fitness;
            if (population[i].fitness > population[bestIdx].fitness) {
                bestIdx = i;
            }
        }
        System.out.printf("[GAv2] gen=%d best=%.0f mean=%.0f (fitness=maxX avg of %d ep)%n",
                generation, population[bestIdx].fitness, sum / populationSize, evalsPerGenome);

        Genome[] next = new Genome[populationSize];

        // 1) 엘리트: 적합도 상위 ELITE_COUNT개를 그대로 보존.
        Integer[] order = sortedByFitnessDesc();
        for (int e = 0; e < ELITE_COUNT; e++) {
            next[e] = population[order[e]].copy();
        }

        // 2) 나머지: 토너먼트로 부모 둘 → 균등 교배 → 돌연변이.
        for (int i = ELITE_COUNT; i < populationSize; i++) {
            Genome child = crossover(tournamentSelect(), tournamentSelect());
            mutate(child);
            next[i] = child;
        }

        population = next;
    }

    /** 적합도 내림차순으로 정렬한 개체 인덱스 배열(엘리트 추출용). */
    private Integer[] sortedByFitnessDesc() {
        Integer[] order = new Integer[populationSize];
        for (int i = 0; i < populationSize; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(population[y].fitness, population[x].fitness));
        return order;
    }

    /** 토너먼트 선택 — 무작위 {@value #TOURNAMENT_SIZE}개 중 적합도 최고 개체. */
    private Genome tournamentSelect() {
        Genome best = population[random.nextInt(populationSize)];
        for (int t = 1; t < TOURNAMENT_SIZE; t++) {
            Genome challenger = population[random.nextInt(populationSize)];
            if (challenger.fitness > best.fitness) {
                best = challenger;
            }
        }
        return best;
    }

    /** 균등 교배 — 각 유전자를 50% 확률로 부모 둘 중 하나에서 물려받는다. */
    private Genome crossover(Genome a, Genome b) {
        Genome child = new Genome();
        for (int s = 0; s < COMMON_SIZE; s++) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                child.qCommon[s][act] = random.nextBoolean() ? a.qCommon[s][act] : b.qCommon[s][act];
            }
        }
        return child;
    }

    /** 돌연변이 — 각 유전자를 {@value #MUTATION_RATE} 확률로 가우시안 노이즈만큼 흔든다. */
    private void mutate(Genome g) {
        for (double[] row : g.qCommon) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                if (random.nextDouble() < MUTATION_RATE) {
                    row[act] += random.nextGaussian() * MUTATION_SIGMA;
                }
            }
        }
    }

    /** 작은 가우시안 노이즈로 채운 새 개체(초기 모집단용). */
    private Genome randomGenome() {
        Genome g = new Genome();
        for (double[] row : g.qCommon) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                row[act] = random.nextGaussian() * INIT_SIGMA;
            }
        }
        return g;
    }

    /** GA는 ε 탐험이 없다 — v2의 탐험은 softmax의 확률성과 돌연변이가 맡는다. 로그엔 0.0으로 보고. */
    @Override
    public double getEpsilon() {
        return 0.0;
    }

    /**
     * 한 개체(정책) — <b>{@code Q_공통} 한 테이블</b>(756개 가중치) + 적합도.
     * [DAY13]의 {@code Q_위치}(7,560개)를 뺀 것이 v2의 처방ⓑ다.
     */
    private static final class Genome {
        final double[][] qCommon = new double[COMMON_SIZE][ACTION_SIZE];
        double fitness;

        /** 엘리트 보존용 깊은 복사. */
        Genome copy() {
            Genome c = new Genome();
            for (int s = 0; s < COMMON_SIZE; s++) {
                System.arraycopy(qCommon[s], 0, c.qCommon[s], 0, ACTION_SIZE);
            }
            c.fitness = fitness;
            return c;
        }
    }
}
