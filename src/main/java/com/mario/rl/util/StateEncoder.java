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
    /** 벽 거리 단계에 곱하는 가중치 (= 구덩이3 × 굼바4 × 방향3 = 36). [DAY8] */
    private static final int WALL_WEIGHT = PIT_DISTS * ENEMY_DISTS * VY_DIRS;
    /** 구덩이 거리 단계에 곱하는 가중치 (= VY_DIRS × 굼바4 = 12). */
    private static final int PIT_WEIGHT = VY_DIRS * ENEMY_DISTS;
    /** 굼바 거리 단계에 곱하는 가중치 (= VY_DIRS). */
    private static final int ENEMY_WEIGHT = VY_DIRS;

    /** Q_공통(위치 무시) 차원 크기 (= 3벽 × 3구덩이 × 4굼바 × 3방향 = 108). [DAY8] 벽 추가로 36→108 */
    public static final int COMMON_SIZE = WALL_DISTS * PIT_DISTS * ENEMY_DISTS * VY_DIRS;
    /** Q_위치(위치 포함) 차원 크기 (= 30구간 × 108 = 3240). */
    public static final int LOCAL_SIZE = X_ZONES * COMMON_SIZE;
    /** xZone에 곱하는 가중치 (= COMMON_SIZE). */
    private static final int X_ZONE_WEIGHT = COMMON_SIZE;

    /**
     * 위치를 무시하고 굼바·구덩이·방향만으로 인코딩한다(`Q_공통`용).
     *
     * @param state 게임 상태
     * @return 0 ~ {@value #COMMON_SIZE}-1
     */
    public int encodeCommon(GameState state) {
        int enemyDist = state.getEnemyDist();   // 0 없음 / 1 멂 / 2 가까움 / 3 위험함
        int pitDist = state.getPitDist();       // 0 없음 / 1 가까움 / 2 코앞
        int wallDist = state.getWallDist();     // 0 없음 / 1 가까움 / 2 코앞 [DAY8]
        int vyDir = state.getVyDir();           // 0 지상·정지 / 1 상승 / 2 하강
        return (wallDist * WALL_WEIGHT) + (pitDist * PIT_WEIGHT) + (enemyDist * ENEMY_WEIGHT) + vyDir;
    }

    /**
     * 위치(xZone)까지 포함해 인코딩한다(`Q_위치`용).
     *
     * @param state 게임 상태
     * @return 0 ~ {@value #LOCAL_SIZE}-1
     */
    public int encodeLocal(GameState state) {
        int xZone = Math.min(state.getMarioX(), X_MAX) / X_ZONE_WIDTH;
        int index = (xZone * X_ZONE_WEIGHT) + encodeCommon(state);
        return Math.min(index, LOCAL_SIZE - 1);
    }
}
