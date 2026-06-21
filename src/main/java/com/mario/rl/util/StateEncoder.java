package com.mario.rl.util;

import com.mario.rl.model.GameState;

/**
 * {@link GameState}를 Q-Table 인덱스(정수)로 변환하는 상태 인코더.
 *
 * <p>연속적인 게임 상태를 이산적인 정수 인덱스로 압축하여 Q-Table의 행(state) 인덱스로 사용한다.
 * 인코딩에 사용하는 특징은 다음 세 가지다:</p>
 * <ul>
 *   <li>마리오의 X 좌표를 {@value #X_ZONES}개 구간으로 분할</li>
 *   <li>앞쪽 굼바까지의 거리 단계 (0 없음 / 1 멂 / 2 가까움 / 3 위험함)</li>
 *   <li>앞쪽 구덩이까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞) — [DAY4]</li>
 *   <li>마리오의 높이를 지상(0)/낮은공중(1)/높은공중·정점(2) 3단계로 — [DAY5 트라이1]</li>
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
     * 높이(yZone) 단계 임계값. {@code mario_y}(= SMB y_pos)는 <b>위로 갈수록 커진다</b>
     * (실측[DAY5]: 지상≈79 · 점프정점 160~199). 값이 클수록 공중에 높이 떠 있다.
     * <ul>
     *   <li>y &lt; {@value #Y_GROUND_MAX} → 0 지상 (점프 안 함/막 떠오름)</li>
     *   <li>{@value #Y_GROUND_MAX} ≤ y &lt; {@value #Y_AIR_MAX} → 1 낮은 공중 (상승·하강 중)</li>
     *   <li>y ≥ {@value #Y_AIR_MAX} → 2 높은 공중·정점</li>
     * </ul>
     */
    private static final int Y_GROUND_MAX = 100;
    private static final int Y_AIR_MAX = 150;
    /** 높이 단계 수 (yZone: 0~2). [DAY5 트라이1] 2→3 */
    private static final int Y_ZONES = 3;
    /** 인덱스 계산 시 X 구간에 곱하는 가중치. (= Y_ZONES × 굼바4 × 구덩이3 = 36) */
    private static final int X_ZONE_WEIGHT = 36;
    /** 인덱스 계산 시 구덩이 거리 단계(0~2)에 곱하는 가중치. (= Y_ZONES × 굼바4 = 12) [DAY4] */
    private static final int PIT_WEIGHT = 12;
    /** 인덱스 계산 시 굼바 거리 단계(0~3)에 곱하는 가중치. (= Y_ZONES) */
    private static final int ENEMY_WEIGHT = Y_ZONES;
    /** Q-Table의 상태 차원 크기 (= 30 × 3구덩이 × 4굼바 × 3높이 = 1080). */
    public static final int STATE_SIZE = X_ZONES * Y_ZONES * 4 * 3;

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
        int marioY = state.getMarioY();         // 위로 갈수록 커짐 (지상≈79·정점 160~199)
        int yZone = marioY < Y_GROUND_MAX ? 0 : (marioY < Y_AIR_MAX ? 1 : 2);

        int index = (xZone * X_ZONE_WEIGHT) + (pitDist * PIT_WEIGHT)
                + (enemyDist * ENEMY_WEIGHT) + yZone;
        return Math.min(index, STATE_SIZE - 1);
    }
}
