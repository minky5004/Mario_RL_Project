package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.util.Random;

/**
 * 유전 알고리즘(GA) — <b>Q값·그래디언트 없이 진화로 정책을 찾는</b> 첫 두뇌. [DAY13]
 *
 * <p>지금까지의 두뇌(QLearning·SARSA·ExpectedSARSA·MonteCarlo)는 <i>한 판 안에서</i> 가치를
 * 조금씩 갱신했다. GA는 다르다 — <b>정책 여러 개(모집단)를 각각 한 판씩 굴려 점수(적합도)를 매기고,
 * 세대 단위로 잘한 개체를 골라 섞고(교배) 흔들어(돌연변이) 다음 세대를 만든다.</b> 스텝별 학습이 없다.</p>
 *
 * <p><b>개체(genome) = 표 두뇌와 똑같은 두 테이블</b> {@code qCommon[108][7] + qLocal[1080][7]}의 실수
 * 가중치다(사용자 결정 — TD/MC와 "완전히 같은 정책 표현"이라 학습 방식만 다른 공정 비교가 된다).
 * 행동은 두 테이블 합산값의 argmax(탐험 ε 없음 — 탐험은 돌연변이가 맡는다).</p>
 *
 * <p><b>적합도 = 그 판 보상 합</b>(사용자 결정). {@link #learn}이 매 스텝 들어오는 reward를 누적하고,
 * {@link #endEpisode}에서 그 값을 현재 개체의 적합도로 확정한다. 전진보상이 {@code +0.1×px}라
 * 사실상 도달 거리(maxX)에 비례하면서 클리어/사망 보너스·페널티까지 반영한다.</p>
 *
 * <p>{@link Brain} 인터페이스에 자연스럽게 들어맞는다({@link RandomBrain}처럼 직접 구현 —
 * 모집단 = 테이블 여러 벌이라 한 벌만 드는 {@link TabularBrain}은 상속하지 않는다):</p>
 * <ul>
 *   <li>{@link #selectAction} = <b>지금 평가 중인 개체</b>의 정책으로 argmax 행동.</li>
 *   <li>{@link #learn} = 스텝 학습 없음, 적합도(보상 합)만 누적.</li>
 *   <li>{@link #endEpisode} = 한 개체 평가 끝 → 적합도 기록 → 다음 개체로. 모집단을 다 돌면 <b>세대 진화</b>.</li>
 * </ul>
 *
 * <p>총 에피소드 예산(1500)을 {@code 모집단 × 세대}로 쪼개 쓴다({@value #POPULATION_SIZE}×50=1500).
 * {@link RLAgent}·소켓·프로토콜은 한 줄도 바뀌지 않는다.</p>
 */
public class GeneticAlgorithm implements Brain {

    /** `Q_공통`(위치 무시) 상태 크기. */
    private static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** `Q_위치`(위치 포함) 상태 크기. */
    private static final int LOCAL_SIZE = StateEncoder.LOCAL_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 모집단 크기 — 한 세대의 개체 수. 1500판 ÷ 30 = 50세대. */
    private static final int POPULATION_SIZE = 30;
    /** 엘리트 수 — 적합도 상위 이만큼은 다음 세대로 <b>그대로</b> 살린다(최고 성능 보존). */
    private static final int ELITE_COUNT = 2;
    /** 토너먼트 선택 크기 — 무작위 이만큼 뽑아 그중 최고를 부모로(클수록 선택압↑). */
    private static final int TOURNAMENT_SIZE = 3;
    /** 돌연변이율 — 자식의 각 유전자(가중치)가 흔들릴 확률. */
    private static final double MUTATION_RATE = 0.05;
    /** 돌연변이 세기 — 흔들 때 더하는 가우시안 노이즈의 표준편차. */
    private static final double MUTATION_SIGMA = 0.5;
    /** 초기화 노이즈 세기 — 0으로 채우면 모든 개체가 똑같아 진화가 안 되므로 작은 가우시안으로 뿌린다. */
    private static final double INIT_SIGMA = 0.1;

    /** 난수 생성기(초기화·선택·교배·돌연변이 공용). */
    private final Random random;
    /** 실제로 쓰는 행동 수(≤ {@link #ACTION_SIZE}). [DAY9 호환] 6=긴 점프 OFF. 비교는 다른 두뇌와 동일하게 6. */
    private final int activeActions;

    /** 현재 세대의 모집단(개체 = genome). */
    private Genome[] population;
    /** 지금 평가 중인 개체의 인덱스(0 ~ {@link #POPULATION_SIZE}-1). */
    private int currentIndex;
    /** 지금 평가 중인 개체가 이번 판에서 쌓은 보상 합(= 적합도 후보). */
    private double currentFitness;
    /** 현재 세대 번호(로그용, 1부터). */
    private int generation;
    /** 지금까지 본 최고 적합도(로그용). */
    private double bestFitnessSoFar;

    /** 시드 없이(비결정적), 전체 행동을 쓰는 기본 생성자. */
    public GeneticAlgorithm() {
        this(new Random(), ACTION_SIZE);
    }

    /** 시드 고정 생성자(시드 N회 비교용). */
    public GeneticAlgorithm(long seed) {
        this(new Random(seed), ACTION_SIZE);
    }

    /** 시드·행동 수 지정 생성자({@code activeActions=6}이면 긴 점프 OFF). */
    public GeneticAlgorithm(long seed, int activeActions) {
        this(new Random(seed), activeActions);
    }

    /** 시드 없이 행동 수만 지정. */
    public GeneticAlgorithm(int activeActions) {
        this(new Random(), activeActions);
    }

    private GeneticAlgorithm(Random random, int activeActions) {
        if (activeActions < 1 || activeActions > ACTION_SIZE) {
            throw new IllegalArgumentException(
                    "activeActions 는 1~" + ACTION_SIZE + " 범위여야 함: " + activeActions);
        }
        this.random = random;
        this.activeActions = activeActions;
        this.population = new Genome[POPULATION_SIZE];
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i] = randomGenome();
        }
        this.currentIndex = 0;
        this.currentFitness = 0.0;
        this.generation = 1;
        this.bestFitnessSoFar = Double.NEGATIVE_INFINITY;
    }

    /** 지금 평가 중인 개체의 정책으로 행동을 고른다(두 테이블 합산 argmax, ε 없음). */
    @Override
    public int selectAction(int common, int local) {
        return argMax(population[currentIndex], common, local);
    }

    /** GA는 스텝마다 학습하지 않는다 — 보상만 적합도로 누적한다(done은 {@link #endEpisode}에서 처리). */
    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        currentFitness += reward;
    }

    /**
     * 한 개체의 평가가 끝났다 — 적합도를 확정하고 다음 개체로 넘긴다.
     * 모집단을 다 평가했으면(인덱스가 한 바퀴) 세대 진화를 수행한다.
     */
    @Override
    public void endEpisode() {
        population[currentIndex].fitness = currentFitness;
        if (currentFitness > bestFitnessSoFar) {
            bestFitnessSoFar = currentFitness;
        }
        currentFitness = 0.0;

        currentIndex++;
        if (currentIndex >= POPULATION_SIZE) {
            evolve();
            currentIndex = 0;
            generation++;
        }
    }

    /**
     * 한 세대를 진화시킨다 — 엘리트 보존 + 토너먼트 선택 + 균등 교배 + 돌연변이.
     *
     * <ol>
     *   <li>적합도 상위 {@value #ELITE_COUNT}개를 그대로 다음 세대로 복사(최고 성능 유실 방지).</li>
     *   <li>나머지는 부모 둘을 각각 토너먼트로 뽑아 균등 교배 → 돌연변이로 자식을 만든다.</li>
     * </ol>
     */
    private void evolve() {
        // 진화 직전 현재 세대 통계 로그(일지/그래프용).
        int bestIdx = 0;
        double sum = 0.0;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            sum += population[i].fitness;
            if (population[i].fitness > population[bestIdx].fitness) {
                bestIdx = i;
            }
        }
        System.out.printf("[GA] gen=%d best=%.1f mean=%.1f%n",
                generation, population[bestIdx].fitness, sum / POPULATION_SIZE);

        Genome[] next = new Genome[POPULATION_SIZE];

        // 1) 엘리트: 적합도 상위 ELITE_COUNT개를 그대로 보존.
        Integer[] order = sortedByFitnessDesc();
        for (int e = 0; e < ELITE_COUNT; e++) {
            next[e] = population[order[e]].copy();
        }

        // 2) 나머지: 토너먼트로 부모 둘 → 균등 교배 → 돌연변이.
        for (int i = ELITE_COUNT; i < POPULATION_SIZE; i++) {
            Genome parentA = tournamentSelect();
            Genome parentB = tournamentSelect();
            Genome child = crossover(parentA, parentB);
            mutate(child);
            next[i] = child;
        }

        population = next;
    }

    /** 적합도 내림차순으로 정렬한 개체 인덱스 배열을 돌려준다(엘리트 추출용). */
    private Integer[] sortedByFitnessDesc() {
        Integer[] order = new Integer[POPULATION_SIZE];
        for (int i = 0; i < POPULATION_SIZE; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(population[y].fitness, population[x].fitness));
        return order;
    }

    /** 토너먼트 선택 — 무작위 {@value #TOURNAMENT_SIZE}개를 뽑아 그중 적합도가 가장 높은 개체를 반환. */
    private Genome tournamentSelect() {
        Genome best = population[random.nextInt(POPULATION_SIZE)];
        for (int t = 1; t < TOURNAMENT_SIZE; t++) {
            Genome challenger = population[random.nextInt(POPULATION_SIZE)];
            if (challenger.fitness > best.fitness) {
                best = challenger;
            }
        }
        return best;
    }

    /** 균등 교배 — 각 유전자를 50% 확률로 부모 둘 중 하나에서 물려받는다. */
    private Genome crossover(Genome a, Genome b) {
        Genome child = new Genome();
        crossTable(child.qCommon, a.qCommon, b.qCommon);
        crossTable(child.qLocal, a.qLocal, b.qLocal);
        return child;
    }

    private void crossTable(double[][] out, double[][] a, double[][] b) {
        for (int s = 0; s < out.length; s++) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                out[s][act] = random.nextBoolean() ? a[s][act] : b[s][act];
            }
        }
    }

    /** 돌연변이 — 각 유전자를 {@value #MUTATION_RATE} 확률로 가우시안 노이즈만큼 흔든다. */
    private void mutate(Genome g) {
        mutateTable(g.qCommon);
        mutateTable(g.qLocal);
    }

    private void mutateTable(double[][] table) {
        for (double[] row : table) {
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
        fillRandom(g.qCommon);
        fillRandom(g.qLocal);
        return g;
    }

    private void fillRandom(double[][] table) {
        for (double[] row : table) {
            for (int act = 0; act < ACTION_SIZE; act++) {
                row[act] = random.nextGaussian() * INIT_SIGMA;
            }
        }
    }

    /** 한 개체의 두 테이블 합산값이 최대인 행동(쓰는 행동 범위 내). 동점이면 더 앞선 인덱스(표 두뇌와 동일). */
    private int argMax(Genome g, int common, int local) {
        int bestIndex = 0;
        double bestValue = g.qCommon[common][0] + g.qLocal[local][0];
        for (int act = 1; act < activeActions; act++) {
            double v = g.qCommon[common][act] + g.qLocal[local][act];
            if (v > bestValue) {
                bestValue = v;
                bestIndex = act;
            }
        }
        return bestIndex;
    }

    /** GA는 ε 탐험이 없다 — 탐험은 돌연변이가 맡는다. 로그·그래프엔 0.0으로 보고한다. */
    @Override
    public double getEpsilon() {
        return 0.0;
    }

    /**
     * 한 개체(정책) — 표 두뇌와 똑같은 두 테이블 가중치 + 적합도.
     * 진화는 이 가중치를 직접 섞고 흔들어 더 나은 정책을 찾는다(Q값 갱신식·부트스트랩 없음).
     */
    private static final class Genome {
        final double[][] qCommon = new double[COMMON_SIZE][ACTION_SIZE];
        final double[][] qLocal = new double[LOCAL_SIZE][ACTION_SIZE];
        double fitness;

        /** 엘리트 보존용 깊은 복사. */
        Genome copy() {
            Genome c = new Genome();
            for (int s = 0; s < COMMON_SIZE; s++) {
                System.arraycopy(qCommon[s], 0, c.qCommon[s], 0, ACTION_SIZE);
            }
            for (int s = 0; s < LOCAL_SIZE; s++) {
                System.arraycopy(qLocal[s], 0, c.qLocal[s], 0, ACTION_SIZE);
            }
            c.fitness = fitness;
            return c;
        }
    }
}
