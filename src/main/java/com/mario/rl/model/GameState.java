package com.mario.rl.model;

/**
 * Python 게임 환경으로부터 수신하는 게임 상태(GameState) 데이터 모델.
 *
 * <p>Python 서버가 매 스텝마다 JSON으로 전송하며, Gson이 이 클래스로 역직렬화한다.
 * 필드명은 Python이 보내는 JSON 키(snake_case)와 정확히 일치해야 한다.</p>
 *
 * <p>수신 JSON 예시:
 * <pre>
 * {
 *   "mario_x": 320, "mario_y": 180, "enemy_near": false,
 *   "score": 100, "time_left": 380, "reward": 3.2,
 *   "done": false, "info": "running"
 * }
 * </pre></p>
 */
public class GameState {

    /** 마리오의 X 좌표 (0~3000 범위). */
    private int mario_x;
    /** 마리오의 Y 좌표. */
    private int mario_y;
    /** 주변 2칸 이내 적 존재 여부. */
    private boolean enemy_near;
    /** 현재 점수. */
    private int score;
    /** 남은 시간. */
    private int time_left;
    /** Python이 계산한 보상값. */
    private double reward;
    /** 에피소드 종료 여부. */
    private boolean done;
    /** 종료 원인 ("dead", "clear", "timeout", "running"). */
    private String info;

    /** 마리오의 X 좌표를 반환한다. */
    public int getMarioX() {
        return mario_x;
    }

    /** 마리오의 X 좌표를 설정한다. */
    public void setMarioX(int mario_x) {
        this.mario_x = mario_x;
    }

    /** 마리오의 Y 좌표를 반환한다. */
    public int getMarioY() {
        return mario_y;
    }

    /** 마리오의 Y 좌표를 설정한다. */
    public void setMarioY(int mario_y) {
        this.mario_y = mario_y;
    }

    /** 주변 적 존재 여부를 반환한다. */
    public boolean isEnemyNear() {
        return enemy_near;
    }

    /** 주변 적 존재 여부를 설정한다. */
    public void setEnemyNear(boolean enemy_near) {
        this.enemy_near = enemy_near;
    }

    /** 현재 점수를 반환한다. */
    public int getScore() {
        return score;
    }

    /** 현재 점수를 설정한다. */
    public void setScore(int score) {
        this.score = score;
    }

    /** 남은 시간을 반환한다. */
    public int getTimeLeft() {
        return time_left;
    }

    /** 남은 시간을 설정한다. */
    public void setTimeLeft(int time_left) {
        this.time_left = time_left;
    }

    /** 보상값을 반환한다. */
    public double getReward() {
        return reward;
    }

    /** 보상값을 설정한다. */
    public void setReward(double reward) {
        this.reward = reward;
    }

    /** 에피소드 종료 여부를 반환한다. */
    public boolean isDone() {
        return done;
    }

    /** 에피소드 종료 여부를 설정한다. */
    public void setDone(boolean done) {
        this.done = done;
    }

    /** 종료 원인 문자열을 반환한다. */
    public String getInfo() {
        return info;
    }

    /** 종료 원인 문자열을 설정한다. */
    public void setInfo(String info) {
        this.info = info;
    }

    @Override
    public String toString() {
        return "GameState{" +
                "mario_x=" + mario_x +
                ", mario_y=" + mario_y +
                ", enemy_near=" + enemy_near +
                ", score=" + score +
                ", time_left=" + time_left +
                ", reward=" + reward +
                ", done=" + done +
                ", info='" + info + '\'' +
                '}';
    }
}
