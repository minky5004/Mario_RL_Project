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
 * Q-Learning 핵심 알고리즘.
 *
 * <p>[DAY7 트라이2] <b>두 개의 Q-Table을 합산</b>해 행동을 고른다(tile/coarse coding):</p>
 * <ul>
 *   <li>{@code qCommon[COMMON_SIZE][A]} — 위치를 무시한 상태(`Q_공통`). 모든 위치의 경험이 합쳐져
 *       "굼바=넘기"를 빠르게 일반화한다.</li>
 *   <li>{@code qLocal[LOCAL_SIZE][A]} — 위치(xZone)까지 포함(`Q_위치`). 특정 지점의 예외를 덧칠해 안정화.</li>
 * </ul>
 *
 * <p>행동 가치는 두 테이블의 합 {@code Q(s,a) = qCommon[c][a] + qLocal[l][a]} 이다.
 * TD 오차 δ는 양쪽에 똑같이, 단 학습률을 타일 수(2)로 나눠 적용한다(합산 Q의 실효 학습률 발산 방지):
 * <pre>
 * target = done ? reward : reward + γ * max_a( qCommon[nc][a] + qLocal[nl][a] )
 * δ = target - (qCommon[c][a] + qLocal[l][a])
 * qCommon[c][a] += (α/2) * δ ;  qLocal[l][a] += (α/2) * δ
 * </pre></p>
 */
public class QLearning {

    /** `Q_공통`(위치 무시) 상태 크기. */
    private static final int COMMON_SIZE = StateEncoder.COMMON_SIZE;
    /** `Q_위치`(위치 포함) 상태 크기. */
    private static final int LOCAL_SIZE = StateEncoder.LOCAL_SIZE;
    /** 행동 차원 크기. */
    private static final int ACTION_SIZE = Action.ACTION_SIZE;

    /** 학습률 (α). */
    private static final double LEARNING_RATE = 0.1;
    /** 테이블별 학습률 — TD 오차를 두 테이블(타일)에 나눠 적용(α/타일수). */
    private static final double TABLE_LR = LEARNING_RATE / 2.0;
    /** 할인율 (γ). */
    private static final double DISCOUNT_FACTOR = 0.99;
    /** epsilon 초기값 (탐험 비율). */
    private static final double INITIAL_EPSILON = 1.0;
    /** 에피소드마다 곱해지는 epsilon 감쇠율. */
    private static final double EPSILON_DECAY = 0.995;
    /** epsilon 하한값. */
    private static final double EPSILON_MIN = 0.01;
    /** 저장 파일 매직 넘버 ("QTBL"). [DAY9 작업3] */
    private static final int MAGIC = 0x5154_424C;

    /** `Q_공통` 테이블: [위치 무시 상태][행동]. */
    private final double[][] qCommon;
    /** `Q_위치` 테이블: [위치 포함 상태][행동]. */
    private final double[][] qLocal;
    /** 난수 생성기 (탐험 및 무작위 동점 처리용). */
    private final Random random;
    /**
     * 실제로 선택·평가하는 행동 수 (≤ {@link #ACTION_SIZE}). [DAY9 트라이1 베이스라인]
     * 7로 두면 긴 점프 포함, 6으로 두면 0~5만 써 긴 점프를 끈다(DAY8과 동일한 6행동).
     * 테이블은 항상 {@code ACTION_SIZE} 폭이라, 안 쓰는 열은 0으로 남아 선택되지 않는다.
     */
    private final int activeActions;
    /** 현재 탐험 비율. */
    private double epsilon;

    /**
     * Q-Learning 에이전트를 생성한다. 두 Q-Table은 0으로 초기화되고 epsilon은 1.0에서 시작한다.
     * 난수는 시드 없이(비결정적), 행동은 전체({@link #ACTION_SIZE})를 쓴다.
     */
    public QLearning() {
        this(new Random(), ACTION_SIZE);
    }

    /**
     * 시드를 고정해 Q-Learning 에이전트를 생성한다. [DAY9 작업1] 시드 N회 반복 비교용 —
     * 같은 시드면 탐험(epsilon-greedy)·동점 처리가 재현돼 실행 분산을 통제할 수 있다.
     *
     * @param seed 난수 시드
     */
    public QLearning(long seed) {
        this(new Random(seed), ACTION_SIZE);
    }

    /**
     * 시드·행동 수를 모두 지정해 생성한다. [DAY9 트라이1 베이스라인]
     * {@code activeActions=6}이면 긴 점프(인덱스 6)를 빼고 학습한다(매칭 베이스라인).
     *
     * @param seed          난수 시드
     * @param activeActions 실제로 쓰는 행동 수 (1 ~ {@link #ACTION_SIZE})
     */
    public QLearning(long seed, int activeActions) {
        this(new Random(seed), activeActions);
    }

    /** 공통 생성자 — 두 Q-Table을 0으로, epsilon을 초기값으로, 주어진 난수원·행동 수로 초기화. */
    private QLearning(Random random, int activeActions) {
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
     * epsilon-greedy 정책으로 행동 인덱스를 선택한다(두 테이블 합산 기준).
     *
     * @param common `Q_공통` 상태 인덱스
     * @param local  `Q_위치` 상태 인덱스
     * @return 선택된 행동 인덱스 (0 ~ ACTION_SIZE-1)
     */
    public int selectAction(int common, int local) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(activeActions);
        }
        return argMaxCombined(common, local);
    }

    /**
     * 두 Q-Table을 동시에 갱신한다. TD 오차를 양쪽에 α/2 씩 적용한다.
     *
     * @param common     현재 `Q_공통` 인덱스
     * @param local      현재 `Q_위치` 인덱스
     * @param action     수행한 행동 인덱스
     * @param reward     받은 보상
     * @param nextCommon 다음 `Q_공통` 인덱스
     * @param nextLocal  다음 `Q_위치` 인덱스
     * @param done       에피소드 종료 여부 (true면 미래 보상을 더하지 않음)
     */
    public void update(int common, int local, int action, double reward,
                       int nextCommon, int nextLocal, boolean done) {
        double q = qCommon[common][action] + qLocal[local][action];
        double target = done
                ? reward
                : reward + DISCOUNT_FACTOR * maxCombined(nextCommon, nextLocal);
        double delta = target - q;
        qCommon[common][action] += TABLE_LR * delta;
        qLocal[local][action] += TABLE_LR * delta;
    }

    /**
     * epsilon을 감쇠시킨다. 하한값({@value #EPSILON_MIN}) 아래로는 내려가지 않는다.
     */
    public void decayEpsilon() {
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    /**
     * 현재 탐험 비율(epsilon)을 반환한다.
     *
     * @return 현재 epsilon 값
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * 두 Q-Table과 epsilon을 파일로 저장한다. [DAY9 작업3]
     *
     * <p>형식(빅엔디안 바이너리): 매직 {@code "QTBL"} → 차원 3개(common·local·action) →
     * epsilon → qCommon → qLocal. 차원을 헤더에 적어 로드 시 현재 코드와 모양이 다르면 거부한다.</p>
     *
     * @param path 저장 경로
     * @throws IOException 파일 쓰기 실패 시
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

    /**
     * 저장된 두 Q-Table과 epsilon을 파일에서 읽어 현재 상태에 덮어쓴다. [DAY9 작업3]
     *
     * @param path 저장 경로
     * @throws IOException        파일 읽기 실패 시
     * @throws IllegalStateException 매직/차원이 현재 코드와 다른 경우(예: 행동 수 변경)
     */
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

    /** 두 테이블 합산값이 최대인 행동 인덱스(쓰는 행동 범위 내). 동점이면 더 앞선 인덱스. */
    private int argMaxCombined(int common, int local) {
        int bestIndex = 0;
        double bestValue = qCommon[common][0] + qLocal[local][0];
        for (int a = 1; a < activeActions; a++) {
            double v = qCommon[common][a] + qLocal[local][a];
            if (v > bestValue) {
                bestValue = v;
                bestIndex = a;
            }
        }
        return bestIndex;
    }

    /** 두 테이블 합산값의 최댓값(쓰는 행동 범위 내). */
    private double maxCombined(int common, int local) {
        double max = qCommon[common][0] + qLocal[local][0];
        for (int a = 1; a < activeActions; a++) {
            double v = qCommon[common][a] + qLocal[local][a];
            if (v > max) {
                max = v;
            }
        }
        return max;
    }
}
