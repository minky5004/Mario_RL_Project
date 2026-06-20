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
 *   "mario_x": 320, "mario_y": 180, "enemy_dist": 2,
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
    /** 앞쪽 굼바까지의 거리 단계 (0 없음 / 1 멂 / 2 가까움 / 3 위험함). */
    private int enemy_dist;
    /** 앞쪽 구덩이까지의 거리 단계 (0 없음 / 1 가까움 / 2 코앞). [DAY4] */
    private int pit_dist;
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

    /** 앞쪽 굼바까지의 거리 단계(0~3)를 반환한다. */
    public int getEnemyDist() {
        return enemy_dist;
    }

    /** 앞쪽 굼바까지의 거리 단계(0~3)를 설정한다. */
    public void setEnemyDist(int enemy_dist) {
        this.enemy_dist = enemy_dist;
    }

    /** 앞쪽 구덩이까지의 거리 단계(0~2)를 반환한다. */
    public int getPitDist() {
        return pit_dist;
    }

    /** 앞쪽 구덩이까지의 거리 단계(0~2)를 설정한다. */
    public void setPitDist(int pit_dist) {
        this.pit_dist = pit_dist;
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
                ", enemy_dist=" + enemy_dist +
                ", pit_dist=" + pit_dist +
                ", score=" + score +
                ", time_left=" + time_left +
                ", reward=" + reward +
                ", done=" + done +
                ", info='" + info + '\'' +
                '}';
    }
}
