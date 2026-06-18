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
 *   <li>마리오의 높이를 높음(0)/낮음(1)으로 이진화</li>
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
    /** 높이 이진화 기준값 (이 값보다 작으면 '높음'). */
    private static final int Y_THRESHOLD = 160;
    /** 인덱스 계산 시 X 구간에 곱하는 가중치. */
    private static final int X_ZONE_WEIGHT = 20;
    /** 인덱스 계산 시 굼바 거리 단계(0~3)에 곱하는 가중치. */
    private static final int ENEMY_WEIGHT = 2;
    /** Q-Table의 상태 차원 크기 (인덱스 상한). */
    public static final int STATE_SIZE = 1000;

    /**
     * 게임 상태를 0 이상 {@value #STATE_SIZE} 미만의 Q-Table 인덱스로 변환한다.
     *
     * @param state 인코딩할 게임 상태
     * @return Q-Table 행 인덱스 (0 ~ STATE_SIZE-1)
     */
    public int encode(GameState state) {
        int xZone = Math.min(state.getMarioX(), X_MAX) / X_ZONE_WIDTH;
        int enemyDist = state.getEnemyDist();   // 0 없음 / 1 멂 / 2 가까움 / 3 위험함
        int yZone = state.getMarioY() < Y_THRESHOLD ? 0 : 1;

        int index = (xZone * X_ZONE_WEIGHT) + (enemyDist * ENEMY_WEIGHT) + yZone;
        return Math.min(index, STATE_SIZE - 1);
    }
}
