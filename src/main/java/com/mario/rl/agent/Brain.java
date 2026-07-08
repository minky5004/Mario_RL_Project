package com.mario.rl.agent;

/**
 * 마리오의 "두뇌" — 상태를 받아 행동을 고르고, 경험으로 학습하는 알고리즘의 공통 인터페이스. [DAY10]
 *
 * <p>소켓 너머 Python(게임·보상·측정)은 누가 행동을 정하는지 모르므로, 이 인터페이스만 구현하면
 * 같은 환경에서 알고리즘을 갈아끼워 비교할 수 있다(Q-Learning·SARSA·Monte Carlo·진화 …).
 * {@link RLAgent}는 이 인터페이스에만 의존한다.</p>
 *
 * <p>학습 루프는 SARSA식(on-policy)으로 통일한다 — 다음 행동 {@code nextAction}을 학습 *전에* 골라
 * {@link #learn}에 함께 넘긴다. off-policy(Q-Learning)는 {@code nextAction}을 무시하고 max를 쓰면 되므로
 * 같은 루프로 둘 다 표현된다.</p>
 */
public interface Brain {

    /**
     * epsilon-greedy 등 정책으로 행동을 고른다.
     *
     * @param common `Q_공통` 상태 인덱스
     * @param local  `Q_위치` 상태 인덱스
     * @return 행동 인덱스
     */
    int selectAction(int common, int local);

    /**
     * 한 스텝의 경험으로 학습한다.
     *
     * @param common     현재 `Q_공통` 인덱스
     * @param local      현재 `Q_위치` 인덱스
     * @param action     수행한 행동
     * @param reward     받은 보상
     * @param nextCommon 다음 `Q_공통` 인덱스
     * @param nextLocal  다음 `Q_위치` 인덱스
     * @param nextAction 다음 상태에서 고른 행동(on-policy SARSA용). {@code done}이면 의미 없음(예: -1).
     * @param done       에피소드 종료 여부
     */
    void learn(int common, int local, int action, double reward,
               int nextCommon, int nextLocal, int nextAction, boolean done);

    /**
     * 에피소드 결과를 알린다 — {@link #endEpisode} 직전에 호출된다. [DAY18]
     *
     * <p>{@link #learn}이 받는 건 상태 <i>인덱스</i>와 보상뿐이라 "이번 판에 얼마나 멀리 갔나"를 알 수 없다.
     * 적합도를 도달 거리로 매기는 진화 두뇌({@link GeneticAlgorithmV2})만 이 값을 쓰고,
     * 나머지 두뇌는 기본 구현(no-op) 그대로 둔다.</p>
     *
     * @param maxX 이번 에피소드에 도달한 최대 x좌표
     */
    default void observeEpisodeOutcome(int maxX) {
        // 기본은 무시 — 보상만으로 학습하는 두뇌(TD·MC·GA·ES·Random)는 이 정보가 필요 없다.
    }

    /** 에피소드가 끝날 때 호출(epsilon 감쇠 등). */
    void endEpisode();

    /** 현재 탐험 비율(로그용). */
    double getEpsilon();
}
