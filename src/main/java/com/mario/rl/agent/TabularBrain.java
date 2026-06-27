package com.mario.rl.agent;

import com.mario.rl.model.Action;
import com.mario.rl.util.StateEncoder;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * 표(테이블) 기반 TD 알고리즘의 공통 베이스. [DAY10]
 *
 * <p>[DAY7] <b>두 개의 Q-Table을 합산</b>한다(tile/coarse coding): 행동 가치
 * {@code Q(s,a) = qCommon[c][a] + qLocal[l][a]}. epsilon-greedy 선택·α/2 분할 갱신·저장/로드까지
 * 여기서 공통으로 처리하고, <b>"다음 상태를 무엇으로 부트스트랩하나"</b> 한 곳만
 * {@link #bootstrap}로 빼서 서브클래스가 정한다:</p>
 * <ul>
 *   <li>{@link QLearning}(off-policy) — 다음 상태 행동들의 <b>최댓값</b>(실제 다음 행동 무시).</li>
 *   <li>{@link SARSA}(on-policy) — 다음 상태에서 <b>실제로 고른 행동</b>의 값.</li>
 * </ul>
 *
 * <pre>
 * target = done ? reward : reward + γ * bootstrap(nc, nl, na)
 * δ = target - (qCommon[c][a] + qLocal[l][a])
 * qCommon[c][a] += (α/2) * δ ;  qLocal[l][a] += (α/2) * δ
 * </pre>
 */
public abstract class TabularBrain implements Brain {

    /** `Q_공통`(위치 무시) 상태 크기. */
    protected static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** `Q_위치`(위치 포함) 상태 크기. */
    protected static final int LOCAL_SIZE = StateEncoder.LOCAL_SIZE;
    /** 행동 차원 크기. */
    protected static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 학습률 (α). */
    private static final double LEARNING_RATE = 0.1;
    /** 테이블별 학습률 — TD 오차를 두 테이블(타일)에 나눠 적용(α/타일수). */
    private static final double TABLE_LR = LEARNING_RATE / 2.0;
    /** 할인율 (γ). [DAY12] Monte Carlo도 return 계산에 쓰므로 protected. */
    protected static final double DISCOUNT_FACTOR = 0.99;
    /** epsilon 초기값 (탐험 비율). */
    private static final double INITIAL_EPSILON = 1.0;
    /** 에피소드마다 곱해지는 epsilon 감쇠율. */
    private static final double EPSILON_DECAY = 0.995;
    /** epsilon 하한값. */
    private static final double EPSILON_MIN = 0.01;
    /** 저장 파일 매직 넘버 ("QTBL"). [DAY9] */
    private static final int MAGIC = 0x5154_424C;

    /** `Q_공통` 테이블: [위치 무시 상태][행동]. */
    protected final double[][] qCommon;
    /** `Q_위치` 테이블: [위치 포함 상태][행동]. */
    protected final double[][] qLocal;
    /** 난수 생성기 (탐험 및 무작위 동점 처리용). */
    protected final Random random;
    /**
     * 실제로 선택·평가하는 행동 수 (≤ {@link #ACTION_SIZE}). [DAY9]
     * 7=긴 점프 포함, 6=긴 점프 OFF. 테이블은 항상 {@code ACTION_SIZE} 폭이라 안 쓰는 열은 0으로 남는다.
     */
    protected final int activeActions;
    /** 현재 탐험 비율. */
    protected double epsilon;

    /** 주어진 난수원·행동 수로 초기화(두 테이블 0, epsilon 1.0). */
    protected TabularBrain(Random random, int activeActions) {
        if (activeActions < 1 || activeActions > ACTION_SIZE) {
            throw new IllegalArgumentException("activeActions 는 1~" + ACTION_SIZE + " 범위여야 함: " + activeActions);
        }
        this.qCommon = new double[COMMON_SIZE][ACTION_SIZE];
        this.qLocal = new double[LOCAL_SIZE][ACTION_SIZE];
        this.random = random;
        this.activeActions = activeActions;
        this.epsilon = INITIAL_EPSILON;
    }

    /**
     * 다음 상태를 부트스트랩할 값 — 알고리즘마다 다른 <b>유일한 차이점</b>.
     *
     * @param nextCommon 다음 `Q_공통` 인덱스
     * @param nextLocal  다음 `Q_위치` 인덱스
     * @param nextAction 다음 상태에서 고른 행동(SARSA용, off-policy는 무시)
     * @return 부트스트랩 값
     */
    protected abstract double bootstrap(int nextCommon, int nextLocal, int nextAction);

    @Override
    public int selectAction(int common, int local) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(activeActions);
        }
        return argMaxCombined(common, local);
    }

    @Override
    public void learn(int common, int local, int action, double reward,
                      int nextCommon, int nextLocal, int nextAction, boolean done) {
        double target = done
                ? reward
                : reward + DISCOUNT_FACTOR * bootstrap(nextCommon, nextLocal, nextAction);
        updateToward(common, local, action, target);
    }

    /**
     * 한 (상태, 행동)의 두 테이블 값을 {@code target}쪽으로 α/2씩 당긴다. [DAY12]
     *
     * <p>TD 갱신({@link #learn})과 Monte Carlo의 return 갱신이 공유하는 갱신식 —
     * 차이는 "target을 무엇으로 주느냐"뿐이다(TD는 {@code reward+γ·bootstrap}, MC는 실제 return G).</p>
     *
     * <pre>
     * δ = target - (qCommon[c][a] + qLocal[l][a])
     * qCommon[c][a] += (α/2)·δ ;  qLocal[l][a] += (α/2)·δ
     * </pre>
     */
    protected void updateToward(int common, int local, int action, double target) {
        double delta = target - (qCommon[common][action] + qLocal[local][action]);
        qCommon[common][action] += TABLE_LR * delta;
        qLocal[local][action] += TABLE_LR * delta;
    }

    @Override
    public void endEpisode() {
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    @Override
    public double getEpsilon() {
        return epsilon;
    }

    /** 두 테이블 합산값 {@code qCommon[c][a] + qLocal[l][a]}. */
    protected double combined(int common, int local, int action) {
        return qCommon[common][action] + qLocal[local][action];
    }

    /** 두 테이블 합산값이 최대인 행동 인덱스(쓰는 행동 범위 내). 동점이면 더 앞선 인덱스. */
    protected int argMaxCombined(int common, int local) {
        int bestIndex = 0;
        double bestValue = combined(common, local, 0);
        for (int a = 1; a < activeActions; a++) {
            double v = combined(common, local, a);
            if (v > bestValue) {
                bestValue = v;
                bestIndex = a;
            }
        }
        return bestIndex;
    }

    /** 두 테이블 합산값의 최댓값(쓰는 행동 범위 내). */
    protected double maxCombined(int common, int local) {
        double max = combined(common, local, 0);
        for (int a = 1; a < activeActions; a++) {
            max = Math.max(max, combined(common, local, a));
        }
        return max;
    }

    /**
     * 두 Q-Table과 epsilon을 파일로 저장한다. [DAY9]
     * 형식: 매직 → 차원 3개 → epsilon → qCommon → qLocal (헤더 차원으로 로드 시 모양 검증).
     */
    public void save(String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(MAGIC);
            out.writeInt(COMMON_SIZE);
            out.writeInt(LOCAL_SIZE);
            out.writeInt(ACTION_SIZE);
            out.writeDouble(epsilon);
            writeTable(out, qCommon);
            writeTable(out, qLocal);
        }
    }

    /** 저장된 두 Q-Table과 epsilon을 읽어 덮어쓴다. 매직/차원이 다르면 거부. [DAY9] */
    public void load(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path)))) {
            if (in.readInt() != MAGIC) {
                throw new IllegalStateException("Q-Table 파일 형식이 아닙니다: " + path);
            }
            int common = in.readInt();
            int local = in.readInt();
            int action = in.readInt();
            if (common != COMMON_SIZE || local != LOCAL_SIZE || action != ACTION_SIZE) {
                throw new IllegalStateException(String.format(
                        "Q-Table 차원 불일치(파일 %d·%d·%d ≠ 코드 %d·%d·%d) — 인코딩/행동이 바뀐 모델입니다.",
                        common, local, action, COMMON_SIZE, LOCAL_SIZE, ACTION_SIZE));
            }
            this.epsilon = in.readDouble();
            readTable(in, qCommon);
            readTable(in, qLocal);
        }
    }

    /** 2차원 테이블을 행 우선으로 쓴다. */
    private static void writeTable(DataOutputStream out, double[][] table) throws IOException {
        for (double[] row : table) {
            for (double v : row) {
                out.writeDouble(v);
            }
        }
    }

    /** 2차원 테이블을 행 우선으로 읽어 채운다(모양은 호출 전 헤더로 검증됨). */
    private static void readTable(DataInputStream in, double[][] table) throws IOException {
        for (double[] row : table) {
            for (int a = 0; a < row.length; a++) {
                row[a] = in.readDouble();
            }
        }
    }
}
