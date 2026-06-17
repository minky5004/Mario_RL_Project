package com.mario.rl.model;

/**
 * 마리오가 취할 수 있는 행동(Action)을 정의하는 열거형.
 *
 * <p>각 행동은 Python 게임 환경의 액션 인덱스와 1:1로 대응되는 정수 {@code value}를 가진다.
 * Java AI가 선택한 행동은 이 {@code value}가 JSON({@code {"action": N}})으로 직렬화되어
 * Python 서버로 전송된다. Q-Table의 행동 차원 크기는 {@link #ACTION_SIZE}로 노출한다.</p>
 */
public enum Action {

    /** 아무것도 하지 않음. */
    NOOP(0),
    /** 오른쪽 이동. */
    RIGHT(1),
    /** 오른쪽 + 점프 (마리오 진행에 가장 중요한 행동). */
    RIGHT_JUMP(2),
    /** 제자리 점프. */
    JUMP(3),
    /** 왼쪽 이동. */
    LEFT(4),
    /** 오른쪽 달리기 + 점프. */
    RIGHT_RUN_JUMP(5);

    /** 전체 행동 개수 (Q-Table의 행동 차원 크기). */
    public static final int ACTION_SIZE = 6;

    /** Python 게임 환경의 액션 인덱스. */
    private final int value;

    Action(int value) {
        this.value = value;
    }

    /**
     * 이 행동에 대응하는 정수 액션 인덱스를 반환한다.
     *
     * @return Python 액션 인덱스 (0~5)
     */
    public int getValue() {
        return value;
    }

    /**
     * 정수 액션 인덱스를 대응하는 {@link Action}으로 변환한다.
     *
     * @param value 액션 인덱스 (0~5)
     * @return 해당하는 {@link Action}
     * @throws IllegalArgumentException 정의되지 않은 인덱스가 주어진 경우
     */
    public static Action fromValue(int value) {
        for (Action action : values()) {
            if (action.value == value) {
                return action;
            }
        }
        throw new IllegalArgumentException("정의되지 않은 액션 인덱스: " + value);
    }
}
