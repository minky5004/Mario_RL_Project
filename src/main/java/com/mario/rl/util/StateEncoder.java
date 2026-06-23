package com.mario.rl.util;

import com.mario.rl.model.GameState;

/**
 * {@link GameState}를 Q-Table 인덱스(정수)로 변환하는 상태 인코더.
 *
 * <p>[DAY7 트라이2] <b>두 테이블(공통 + 위치)</b>을 위해 인코딩을 둘로 제공한다:</p>
 * <ul>
 *   <li>{@link #encodeCommon} — 위치를 무시하고 굼바·구덩이·방향만 (0~{@value #COMMON_SIZE}-1).
 *       모든 위치의 경험이 한 칸에 합쳐져 "굼바=넘기"를 위치 독립으로 빠르게 일반화한다(`Q_공통`).</li>
 *   <li>{@link #encodeLocal} — 위치(xZone)까지 포함 (0~{@value #LOCAL_SIZE}-1).
 *       특정 지점의 예외를 덧칠해 안정화한다(`Q_위치`).</li>
 * </ul>
 *
 * <p>트라이1(위치 완전 제거, 36칸)은 일반화로 첫 클리어를 냈으나 ε 고정 후 진동했다.
 * 트라이2는 공통으로 일반화하고 위치로 보정해 "빠른 일반화 + 안정성"을 노린다.</p>
 *
 * <p>공통 특징 네 가지:</p>
 * <ul>
 *   <li>앞쪽 굼바까지의 거리 단계 (0 없음 / 1 멂 / 2 가까움 / 3 위험함)</li>
 *   <li>앞쪽 구덩이까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞) — [DAY4]</li>
 *   <li>앞쪽 벽(토관/계단)까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞) — [DAY8]</li>
 *   <li>마리오의 수직 이동 방향 (0 지상·정지 / 1 상승 / 2 하강) — [DAY5 트라이3]</li>
 * </ul>
 */
public class StateEncoder {

    /** 마리오 X 좌표의 최댓값 (이 값으로 클램프). */
    private static final int X_MAX = 2999;
    /** X 좌표를 분할할 구간 수. */
    private static final int X_ZONES = 30;
    /** 각 X 구간의 폭 ((X_MAX+1) / X_ZONES = 100). */
    private static final int X_ZONE_WIDTH = 100;

    /** 수직 이동 방향 단계 수 (0 지상·정지 / 1 상승 / 2 하강). [DAY5 트라이3] */
    private static final int VY_DIRS = 3;
    /** 굼바 거리 단계 수 (0 없음 / 1 멂 / 2 가까움 / 3 위험함). */
    private static final int ENEMY_DISTS = 4;
    /** 구덩이 거리 단계 수 (0 없음 / 1 가까움 / 2 코앞). [DAY4] */
    private static final int PIT_DISTS = 3;
    /** 벽(토관/계단) 거리 단계 수 (0 없음 / 1 가까움 / 2 코앞). [DAY8] */
    private static final int WALL_DISTS = 3;
    /** 위치 테이블의 칸당 공통부 크기 (= 구덩이3 × 굼바4 × 방향3 = 36). 벽 제외 — 아래 참고. [DAY8 트라이2] */
    private static final int LOCAL_FEATURE_SIZE = PIT_DISTS * ENEMY_DISTS * VY_DIRS;
    /** 구덩이 거리 단계에 곱하는 가중치 (= VY_DIRS × 굼바4 = 12). */
    private static final int PIT_WEIGHT = VY_DIRS * ENEMY_DISTS;
    /** 굼바 거리 단계에 곱하는 가중치 (= VY_DIRS). */
    private static final int ENEMY_WEIGHT = VY_DIRS;
    /** 벽 거리 단계에 곱하는 가중치 (= LOCAL_FEATURE_SIZE = 36, 벽이 공통의 최상위 자리). [DAY8] */
    private static final int WALL_WEIGHT = LOCAL_FEATURE_SIZE;

    /** Q_공통(위치 무시) 차원 크기 (= 3벽 × 36 = 108). [DAY8] 벽 포함 — 토관은 위치 독립이라 여기서 학습. */
    public static final int COMMON_SIZE = WALL_DISTS * LOCAL_FEATURE_SIZE;
    /**
     * Q_위치(위치 포함) 차원 크기 (= 30구간 × 36 = 1080). [DAY8 트라이2]
     *
     * <p>벽은 <b>일부러 뺐다</b>. 토관은 위치와 무관하게 "넘는 것"(위치 독립)이라 {@code Q_공통}만으로 충분하고,
     * {@code Q_위치}에까지 넣으면 위치 테이블이 3배(1080→3240)로 커져 예산만 먹는다(트라이1의 후퇴 원인).
     * 그래서 위치 테이블은 굼바·구덩이·방향(36)만 유지해 DAY7과 같은 1080으로 되돌린다.</p>
     */
    public static final int LOCAL_SIZE = X_ZONES * LOCAL_FEATURE_SIZE;
    /** xZone에 곱하는 가중치 (= LOCAL_FEATURE_SIZE, 벽 제외). */
    private static final int X_ZONE_WEIGHT = LOCAL_FEATURE_SIZE;

    /**
     * 위치·벽을 뺀 공통부 — 굼바·구덩이·방향만 (0 ~ {@value #LOCAL_FEATURE_SIZE}-1).
     * {@link #encodeCommon}과 {@link #encodeLocal}이 공유한다.
     *
     * @param state 게임 상태
     * @return 0 ~ {@value #LOCAL_FEATURE_SIZE}-1
     */
    private int localFeature(GameState state) {
        int enemyDist = state.getEnemyDist();   // 0 없음 / 1 멂 / 2 가까움 / 3 위험함
        int pitDist = state.getPitDist();       // 0 없음 / 1 가까움 / 2 코앞
        int vyDir = state.getVyDir();           // 0 지상·정지 / 1 상승 / 2 하강
        return (pitDist * PIT_WEIGHT) + (enemyDist * ENEMY_WEIGHT) + vyDir;
    }

    /**
     * 위치를 무시하고 굼바·구덩이·방향·<b>벽</b>으로 인코딩한다(`Q_공통`용).
     * 토관(벽)은 위치 독립 신호라 여기(공통)에만 넣는다. [DAY8 트라이2]
     *
     * @param state 게임 상태
     * @return 0 ~ {@value #COMMON_SIZE}-1
     */
    public int encodeCommon(GameState state) {
        int wallDist = state.getWallDist();     // 0 없음 / 1 가까움 / 2 코앞 [DAY8]
        return (wallDist * WALL_WEIGHT) + localFeature(state);
    }

    /**
     * 위치(xZone) + 굼바·구덩이·방향으로 인코딩한다(`Q_위치`용). <b>벽은 제외</b>(위치 독립이라 공통에만). [DAY8 트라이2]
     *
     * @param state 게임 상태
     * @return 0 ~ {@value #LOCAL_SIZE}-1
     */
    public int encodeLocal(GameState state) {
        int xZone = Math.min(state.getMarioX(), X_MAX) / X_ZONE_WIDTH;
        int index = (xZone * X_ZONE_WEIGHT) + localFeature(state);
        return Math.min(index, LOCAL_SIZE - 1);
    }
}
