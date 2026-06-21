package com.mario.rl.util;

import com.mario.rl.model.GameState;

/**
 * {@link GameState}를 Q-Table 인덱스(정수)로 변환하는 상태 인코더.
 *
 * <p>연속적인 게임 상태를 이산적인 정수 인덱스로 압축하여 Q-Table의 행(state) 인덱스로 사용한다.
 * 인코딩에 사용하는 특징은 다음 네 가지다:</p>
 * <ul>
 *   <li>마리오의 X 좌표를 {@value #X_ZONES}개 구간으로 분할</li>
 *   <li>앞쪽 굼바까지의 거리 단계 (0 없음 / 1 멂 / 2 가까움 / 3 위험함)</li>
 *   <li>앞쪽 구덩이까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞) — [DAY4]</li>
 *   <li>마리오의 수직 이동 방향 (0 지상·정지 / 1 상승 / 2 하강) — [DAY5 트라이3, 높이 대체]</li>
 * </ul>
 *
 * <p>반환 인덱스는 절대 {@value #STATE_SIZE} 이상이 되지 않도록 {@link Math#min} 으로 제한한다.</p>
 */
public class StateEncoder {

    /** 마리오 X 좌표의 최댓값 (이 값으로 클램프). */
    private static final int X_MAX = 2999;
    /** X 좌표를 분할할 구간 수. */
    private static final int X_ZONES = 30;
    /** 각 X 구간의 폭 ((X_MAX+1) / X_ZONES = 100). */
    private static final int X_ZONE_WIDTH = 100;
    /**
     * 수직 이동 방향 단계 수 (vyDir: 0 지상·정지 / 1 상승 / 2 하강). [DAY5 트라이3]
     * 같은 높이를 상승·하강 때 두 번 지나는 모호함(yZone의 한계)을 방향으로 해소한다.
     * Python이 직전 y_pos와의 차분(Δy)으로 계산해 보낸다.
     */
    private static final int VY_DIRS = 3;
    /** 인덱스 계산 시 X 구간에 곱하는 가중치. (= VY_DIRS × 굼바4 × 구덩이3 = 36) */
    private static final int X_ZONE_WEIGHT = 36;
    /** 인덱스 계산 시 구덩이 거리 단계(0~2)에 곱하는 가중치. (= VY_DIRS × 굼바4 = 12) [DAY4] */
    private static final int PIT_WEIGHT = 12;
    /** 인덱스 계산 시 굼바 거리 단계(0~3)에 곱하는 가중치. (= VY_DIRS) */
    private static final int ENEMY_WEIGHT = VY_DIRS;
    /** Q-Table의 상태 차원 크기 (= 30 × 3구덩이 × 4굼바 × 3방향 = 1080). */
    public static final int STATE_SIZE = X_ZONES * VY_DIRS * 4 * 3;

    /**
     * 게임 상태를 0 이상 {@value #STATE_SIZE} 미만의 Q-Table 인덱스로 변환한다.
     *
     * @param state 인코딩할 게임 상태
     * @return Q-Table 행 인덱스 (0 ~ STATE_SIZE-1)
     */
    public int encode(GameState state) {
        int xZone = Math.min(state.getMarioX(), X_MAX) / X_ZONE_WIDTH;
        int enemyDist = state.getEnemyDist();   // 0 없음 / 1 멂 / 2 가까움 / 3 위험함
        int pitDist = state.getPitDist();       // 0 없음 / 1 가까움 / 2 코앞 [DAY4]
        int vyDir = state.getVyDir();           // 0 지상·정지 / 1 상승 / 2 하강 [DAY5 트라이3]

        int index = (xZone * X_ZONE_WEIGHT) + (pitDist * PIT_WEIGHT)
                + (enemyDist * ENEMY_WEIGHT) + vyDir;
        return Math.min(index, STATE_SIZE - 1);
    }
}
