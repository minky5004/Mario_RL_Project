package com.mario.rl.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * {@link Action}의 정수 매핑을 검증한다.
 *
 * <p>이 enum의 {@code value}는 <b>Python 게임 서버의 액션 인덱스와 1:1</b>이라,
 * 값이 어긋나면 자바는 "오른쪽 점프"를 골랐는데 게임은 딴 버튼을 누르는 식으로
 * 조용히 학습이 망가진다(에러도 안 난다). 계약이라 못박아 둔다.</p>
 */
class ActionTest {

    @Test
    @DisplayName("[DAY9] 행동은 7개 (긴 점프 추가로 6 → 7)")
    void 행동_개수() {
        assertEquals(7, Action.ACTION_SIZE);
        assertEquals(Action.ACTION_SIZE, Action.values().length,
                "ACTION_SIZE와 실제 enum 개수가 어긋나면 Q-Table 행동 차원이 깨진다");
    }

    @Test
    @DisplayName("액션 인덱스는 0~6이 빠짐없이·중복 없이 쓰인다")
    void 인덱스_연속성() {
        boolean[] 사용됨 = new boolean[Action.ACTION_SIZE];
        for (Action action : Action.values()) {
            int v = action.getValue();
            assertEquals(true, v >= 0 && v < Action.ACTION_SIZE, "범위 밖 인덱스: " + action + "=" + v);
            assertEquals(false, 사용됨[v], "인덱스 중복: " + action + "=" + v);
            사용됨[v] = true;
        }
    }

    @Test
    @DisplayName("fromValue ↔ getValue 왕복이 항상 같은 행동을 돌려준다")
    void 왕복_변환() {
        for (Action action : Action.values()) {
            assertSame(action, Action.fromValue(action.getValue()));
        }
    }

    @Test
    @DisplayName("선언 순서(ordinal)와 액션 인덱스가 일치한다")
    void ordinal_일치() {
        // Q-Table은 행동을 인덱스로 다루므로 둘이 어긋나면 헷갈리기 쉽다.
        for (Action action : Action.values()) {
            assertEquals(action.ordinal(), action.getValue(), action + "의 ordinal과 value 불일치");
        }
    }

    @Test
    @DisplayName("정의되지 않은 인덱스는 예외로 거른다")
    void 잘못된_인덱스() {
        assertThrows(IllegalArgumentException.class, () -> Action.fromValue(-1));
        assertThrows(IllegalArgumentException.class, () -> Action.fromValue(Action.ACTION_SIZE));
        assertThrows(IllegalArgumentException.class, () -> Action.fromValue(99));
    }

    @Test
    @DisplayName("주요 행동의 인덱스는 Python 서버와의 계약값 그대로다")
    void 계약값_고정() {
        assertEquals(0, Action.NOOP.getValue());
        assertEquals(1, Action.RIGHT.getValue());
        assertEquals(2, Action.RIGHT_JUMP.getValue());
        assertEquals(6, Action.RIGHT_LONG_JUMP.getValue());
    }
}
