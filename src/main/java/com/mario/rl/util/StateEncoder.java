package com.mario.rl.util;

import com.mario.rl.model.GameState;

/**
 * {@link GameState}를 Q-Table 인덱스(정수)로 변환하는 상태 인코더.
 *
 * <p>연속적인 게임 상태를 이산적인 정수 인덱스로 압축하여 Q-Table의 행(state) 인덱스로 사용한다.
 * 인코딩에 사용하는 특징은 다음 세 가지다:</p>
 * <ul>
 *   <li>앞쪽 굼바까지의 거리 단계 (0 없음 / 1 멂 / 2 가까움 / 3 위험함)</li>
 *   <li>앞쪽 구덩이까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞) — [DAY4]</li>
 *   <li>마리오의 수직 이동 방향 (0 지상·정지 / 1 상승 / 2 하강) — [DAY5 트라이3]</li>
 * </ul>
 *
 * <p>[DAY7 트라이1] <b>위치(xZone)를 인코딩에서 제거</b>했다. 같은 "굼바 가까움"이라도 x290과
 * x1800이 서로 다른 칸이라 학습이 전이되지 않던 문제(위치별 암기 → 일반화 실패)를 없앤다.
 * 이제 "굼바 가까움"은 위치와 무관하게 한 칸이라, 한 번 배운 점프가 모든 위치에 적용된다.
 * 부수 효과로 상태 수가 1080 → {@value #STATE_SIZE}로 줄어 같은 판수에 각 칸을 훨씬 촘촘히 학습한다.</p>
 *
 * <p>반환 인덱스는 절대 {@value #STATE_SIZE} 이상이 되지 않도록 {@link Math#min} 으로 제한한다.</p>
 */
public class StateEncoder {

    /**
     * 수직 이동 방향 단계 수 (vyDir: 0 지상·정지 / 1 상승 / 2 하강). [DAY5 트라이3]
     * 같은 높이를 상승·하강 때 두 번 지나는 모호함(yZone의 한계)을 방향으로 해소한다.
     * Python이 직전 y_pos와의 차분(Δy)으로 계산해 보낸다.
     */
    private static final int VY_DIRS = 3;
    /** 굼바 거리 단계 수 (0 없음 / 1 멂 / 2 가까움 / 3 위험함). */
    private static final int ENEMY_DISTS = 4;
    /** 구덩이 거리 단계 수 (0 없음 / 1 가까움 / 2 코앞). [DAY4] */
    private static final int PIT_DISTS = 3;
    /** 인덱스 계산 시 구덩이 거리 단계에 곱하는 가중치. (= VY_DIRS × 굼바4 = 12) [DAY4] */
    private static final int PIT_WEIGHT = VY_DIRS * ENEMY_DISTS;
    /** 인덱스 계산 시 굼바 거리 단계에 곱하는 가중치. (= VY_DIRS) */
    private static final int ENEMY_WEIGHT = VY_DIRS;
    /** Q-Table의 상태 차원 크기 (= 3구덩이 × 4굼바 × 3방향 = 36). [DAY7 트라이1] 위치 제거로 1080→36. */
    public static final int STATE_SIZE = PIT_DISTS * ENEMY_DISTS * VY_DIRS;

    /**
     * 게임 상태를 0 이상 {@value #STATE_SIZE} 미만의 Q-Table 인덱스로 변환한다.
     *
     * <p>[DAY7 트라이1] 마리오 X 좌표는 더 이상 인덱스에 쓰지 않는다(위치 독립 일반화).</p>
     *
     * @param state 인코딩할 게임 상태
     * @return Q-Table 행 인덱스 (0 ~ STATE_SIZE-1)
     */
    public int encode(GameState state) {
        int enemyDist = state.getEnemyDist();   // 0 없음 / 1 멂 / 2 가까움 / 3 위험함
        int pitDist = state.getPitDist();       // 0 없음 / 1 가까움 / 2 코앞 [DAY4]
        int vyDir = state.getVyDir();           // 0 지상·정지 / 1 상승 / 2 하강 [DAY5 트라이3]

        int index = (pitDist * PIT_WEIGHT) + (enemyDist * ENEMY_WEIGHT) + vyDir;
        return Math.min(index, STATE_SIZE - 1);
    }
}
