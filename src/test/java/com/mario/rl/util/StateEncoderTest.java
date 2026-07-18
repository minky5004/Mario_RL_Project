package com.mario.rl.util;

import com.mario.rl.model.GameState;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link StateEncoder}의 두 인코딩(`Q_공통` 108칸 / `Q_위치` 1080칸)을 검증한다.
 *
 * <p>인코딩은 자리값(positional) 방식이라 "특징 조합 → 인덱스"가 <b>단사(injective)</b>여야 한다.
 * 두 상태가 같은 칸에 겹치면 서로의 학습을 덮어써 조용히 망가지므로, 전수 조합으로 확인한다.</p>
 *
 * <p>특히 [DAY8 트라이2]의 설계 결정 — <b>벽(토관)은 `Q_공통`에만 넣고 `Q_위치`에서는 뺀다</b> — 는
 * 코드만 봐서는 놓치기 쉬운 의도라 회귀 테스트로 못박는다.</p>
 */
class StateEncoderTest {

    /** 각 특징의 단계 수 (StateEncoder의 private 상수와 같아야 한다). */
    private static final int WALL_DISTS = 3;
    private static final int PIT_DISTS = 3;
    private static final int ENEMY_DISTS = 4;
    private static final int VY_DIRS = 3;
    private static final int X_ZONES = 30;

    private final StateEncoder encoder = new StateEncoder();

    /** 테스트용 상태 하나를 만든다. */
    private static GameState state(int marioX, int enemyDist, int pitDist, int wallDist, int vyDir) {
        GameState s = new GameState();
        s.setMarioX(marioX);
        s.setEnemyDist(enemyDist);
        s.setPitDist(pitDist);
        s.setWallDist(wallDist);
        s.setVyDir(vyDir);
        return s;
    }

    @Test
    @DisplayName("테이블 차원은 108(공통) / 1080(위치)")
    void 테이블_차원() {
        assertEquals(WALL_DISTS * PIT_DISTS * ENEMY_DISTS * VY_DIRS, StateEncoder.COMMON_SIZE);
        assertEquals(108, StateEncoder.COMMON_SIZE);
        assertEquals(X_ZONES * PIT_DISTS * ENEMY_DISTS * VY_DIRS, StateEncoder.LOCAL_SIZE);
        assertEquals(1080, StateEncoder.LOCAL_SIZE);
    }

    @Test
    @DisplayName("encodeCommon: 전체 조합이 0~107을 겹침 없이 정확히 채운다")
    void encodeCommon_전수_단사() {
        Set<Integer> seen = new HashSet<>();
        for (int wall = 0; wall < WALL_DISTS; wall++) {
            for (int pit = 0; pit < PIT_DISTS; pit++) {
                for (int enemy = 0; enemy < ENEMY_DISTS; enemy++) {
                    for (int vy = 0; vy < VY_DIRS; vy++) {
                        int index = encoder.encodeCommon(state(0, enemy, pit, wall, vy));
                        assertTrue(index >= 0 && index < StateEncoder.COMMON_SIZE,
                                "인덱스가 범위를 벗어남: " + index);
                        assertTrue(seen.add(index),
                                "인덱스 충돌: wall=" + wall + " pit=" + pit + " enemy=" + enemy + " vy=" + vy);
                    }
                }
            }
        }
        // 겹침 없이 전부 채웠다면 개수가 곧 차원 크기 (= 빈칸도 없음)
        assertEquals(StateEncoder.COMMON_SIZE, seen.size());
    }

    @Test
    @DisplayName("encodeLocal: 전체 조합이 0~1079를 겹침 없이 정확히 채운다")
    void encodeLocal_전수_단사() {
        Set<Integer> seen = new HashSet<>();
        for (int zone = 0; zone < X_ZONES; zone++) {
            int marioX = zone * 100;   // 각 구간의 시작 좌표
            for (int pit = 0; pit < PIT_DISTS; pit++) {
                for (int enemy = 0; enemy < ENEMY_DISTS; enemy++) {
                    for (int vy = 0; vy < VY_DIRS; vy++) {
                        int index = encoder.encodeLocal(state(marioX, enemy, pit, 0, vy));
                        assertTrue(index >= 0 && index < StateEncoder.LOCAL_SIZE,
                                "인덱스가 범위를 벗어남: " + index);
                        assertTrue(seen.add(index),
                                "인덱스 충돌: zone=" + zone + " pit=" + pit + " enemy=" + enemy + " vy=" + vy);
                    }
                }
            }
        }
        assertEquals(StateEncoder.LOCAL_SIZE, seen.size());
    }

    @Test
    @DisplayName("[DAY8 트라이2] 벽은 공통에만 반영되고 위치 테이블은 무시한다")
    void 벽은_공통에만() {
        GameState 벽없음 = state(500, 1, 1, 0, 1);
        GameState 벽코앞 = state(500, 1, 1, 2, 1);

        // 공통: 벽이 달라지면 다른 칸 (토관을 위치 독립으로 학습)
        assertNotEquals(encoder.encodeCommon(벽없음), encoder.encodeCommon(벽코앞));
        // 위치: 벽은 애초에 안 들어가므로 같은 칸 (1080 유지, 3240으로 안 부풀림)
        assertEquals(encoder.encodeLocal(벽없음), encoder.encodeLocal(벽코앞));
    }

    @Test
    @DisplayName("encodeLocal: xZone은 100px 단위로 끊긴다")
    void xZone_경계() {
        int 구간0끝 = encoder.encodeLocal(state(99, 0, 0, 0, 0));
        int 구간1시작 = encoder.encodeLocal(state(100, 0, 0, 0, 0));
        assertEquals(0, 구간0끝);
        assertNotEquals(구간0끝, 구간1시작);
        // 한 구간(36칸)만큼 정확히 건너뛴다
        assertEquals(36, 구간1시작 - 구간0끝);
    }

    @Test
    @DisplayName("encodeLocal: X_MAX(2999)를 넘는 좌표도 마지막 구간으로 클램프된다")
    void x_클램프() {
        int 마지막구간 = encoder.encodeLocal(state(2999, 0, 0, 0, 0));
        assertEquals(마지막구간, encoder.encodeLocal(state(3000, 0, 0, 0, 0)));
        assertEquals(마지막구간, encoder.encodeLocal(state(999_999, 0, 0, 0, 0)));
        assertTrue(마지막구간 < StateEncoder.LOCAL_SIZE);
    }

    @Test
    @DisplayName("공통 인코딩은 위치를 전혀 보지 않는다")
    void 공통은_위치무시() {
        GameState 출발선 = state(40, 2, 1, 1, 0);
        GameState 깃발앞 = state(2900, 2, 1, 1, 0);
        assertEquals(encoder.encodeCommon(출발선), encoder.encodeCommon(깃발앞));
        assertNotEquals(encoder.encodeLocal(출발선), encoder.encodeLocal(깃발앞));
    }
}
