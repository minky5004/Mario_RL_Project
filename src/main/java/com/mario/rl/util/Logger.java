package com.mario.rl.util;

import com.mario.rl.model.Action;

/**
 * 학습 진행 상황을 콘솔에 출력하는 로거.
 *
 * <p>에피소드 단위 요약 로그({@link #logEpisode})와 스텝 단위 상세 로그({@link #logStep})를 제공한다.
 * 스텝 로그는 출력량이 많으므로 {@link #verbose} 플래그로 제어하며 기본값은 비활성이다.</p>
 */
public class Logger {

    /** 구분선을 출력할 에피소드 주기. */
    private static final int SEPARATOR_INTERVAL = 10;
    /** 구분선 문자열. */
    private static final String SEPARATOR =
            "--------------------------------------------------------------";

    /** 스텝 단위 상세 로그 활성화 여부 (기본 비활성). */
    private boolean verbose = false;

    /**
     * 한 에피소드의 요약 정보를 출력한다.
     *
     * <p>출력 예: {@code [Episode  42] Total Reward:  1243.5 | Max X:  1820 | Epsilon: 0.812}</p>
     * <p>10 에피소드마다 구분선을 함께 출력한다.</p>
     *
     * @param episode     에피소드 번호
     * @param totalReward 해당 에피소드 누적 보상
     * @param maxX        해당 에피소드에서 도달한 최대 X 좌표
     * @param epsilon     현재 탐험 비율(epsilon)
     */
    public void logEpisode(int episode, double totalReward, int maxX, double epsilon) {
        System.out.printf(
                "[Episode %4d] Total Reward: %8.1f | Max X: %5d | Epsilon: %.3f%n",
                episode, totalReward, maxX, epsilon);

        if (episode % SEPARATOR_INTERVAL == 0) {
            System.out.println(SEPARATOR);
        }
    }

    /**
     * 한 스텝의 상세 정보를 출력한다. {@link #verbose}가 {@code true}일 때만 출력된다.
     *
     * @param step   스텝 번호
     * @param action 선택된 행동
     * @param reward 해당 스텝의 보상
     */
    public void logStep(int step, Action action, double reward) {
        if (!verbose) {
            return;
        }
        System.out.printf("  step %4d | action: %-14s | reward: %7.2f%n",
                step, action, reward);
    }

    /**
     * 스텝 단위 상세 로그 활성화 여부를 설정한다.
     *
     * @param verbose {@code true}면 {@link #logStep} 출력 활성화
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}
