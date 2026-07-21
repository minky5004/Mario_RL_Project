package com.mario.rl.agent;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Q-Table 저장/로드({@code -Dmario.save} / {@code -Dmario.load}) 검증. [DAY9]
 *
 * <p>학습 한 판이 시드당 십수 분~몇 시간이라, 저장이 조용히 깨지면 <b>그 시간을 통째로 잃는다</b>.
 * 게다가 인코딩이 바뀐 뒤([DAY4] 240→720, [DAY7] →1080, [DAY8] →108·1080) 옛 파일을 로드하면
 * 차원이 안 맞는 채로 학습이 진행될 수 있어, 헤더 차원 검증이 실제 안전장치다.</p>
 */
class QTablePersistenceTest {

    /** 저장 파일 매직 넘버("QTBL") — {@link TabularBrain}의 private 상수와 같아야 한다. */
    private static final int MAGIC = 0x5154_424C;

    @TempDir
    Path tempDir;

    /** 두뇌에 가짜 경험을 먹여 테이블을 0에서 벗어나게 하고 epsilon도 감쇠시킨다. */
    private static void 학습시킨다(TabularBrain brain, long seed) {
        Random rng = new Random(seed);
        for (int ep = 0; ep < 5; ep++) {
            for (int t = 0; t < 20; t++) {
                brain.learn(
                        rng.nextInt(TabularBrain.COMMON_SIZE), rng.nextInt(TabularBrain.LOCAL_SIZE),
                        rng.nextInt(TabularBrain.ACTION_SIZE), rng.nextDouble() * 10 - 5,
                        rng.nextInt(TabularBrain.COMMON_SIZE), rng.nextInt(TabularBrain.LOCAL_SIZE),
                        rng.nextInt(TabularBrain.ACTION_SIZE), t == 19);
            }
            brain.endEpisode();
        }
    }

    @Test
    @DisplayName("저장 → 로드 왕복이 두 테이블과 epsilon을 그대로 복원한다")
    void 왕복_복원() throws IOException {
        QLearning 원본 = new QLearning(42L, 7);
        학습시킨다(원본, 1L);

        Path file = tempDir.resolve("q-table.bin");
        원본.save(file.toString());
        assertTrue(Files.size(file) > 0, "저장 파일이 비었다");

        QLearning 복원 = new QLearning(42L, 7);
        assertNotEquals(원본.getEpsilon(), 복원.getEpsilon(), "복원 전엔 달라야 검사가 의미 있다");

        복원.load(file.toString());

        for (int c = 0; c < TabularBrain.COMMON_SIZE; c++) {
            assertArrayEquals(원본.qCommon[c], 복원.qCommon[c], 0.0, "qCommon[" + c + "] 불일치");
        }
        for (int l = 0; l < TabularBrain.LOCAL_SIZE; l++) {
            assertArrayEquals(원본.qLocal[l], 복원.qLocal[l], 0.0, "qLocal[" + l + "] 불일치");
        }
        assertEquals(원본.getEpsilon(), 복원.getEpsilon(), 0.0, "epsilon 불일치");
    }

    @Test
    @DisplayName("다른 알고리즘끼리도 파일 형식이 호환된다(이어 학습용)")
    void 알고리즘간_호환() throws IOException {
        // 형식은 TabularBrain 공통이라 QL로 저장한 표를 SARSA(λ)가 이어받을 수 있어야 한다.
        QLearning 저장 = new QLearning(42L, 7);
        학습시킨다(저장, 2L);
        Path file = tempDir.resolve("cross.bin");
        저장.save(file.toString());

        SARSALambda 로드 = new SARSALambda(42L, 7, 0.9);
        로드.load(file.toString());

        assertArrayEquals(저장.qCommon[0], 로드.qCommon[0], 0.0);
        assertEquals(저장.getEpsilon(), 로드.getEpsilon(), 0.0);
    }

    @Test
    @DisplayName("Q-Table 파일이 아니면 거부한다")
    void 매직_불일치_거부() throws IOException {
        Path file = tempDir.resolve("not-a-table.bin");
        Files.write(file, "이건 그냥 텍스트 파일이다".getBytes());

        QLearning brain = new QLearning(42L, 7);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> brain.load(file.toString()));
        assertTrue(e.getMessage().contains("형식"), "형식 오류라고 알려줘야 한다: " + e.getMessage());
    }

    @Test
    @DisplayName("차원이 다른(옛 인코딩) 파일은 거부한다")
    void 차원_불일치_거부() throws IOException {
        // [DAY7] 이전의 720칸짜리 위치 테이블을 흉내낸 헤더
        Path file = tempDir.resolve("old-encoding.bin");
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(file.toFile()))) {
            out.writeInt(MAGIC);
            out.writeInt(TabularBrain.COMMON_SIZE);
            out.writeInt(720);                       // ← 옛 차원
            out.writeInt(TabularBrain.ACTION_SIZE);
            out.writeDouble(0.5);
        }

        QLearning brain = new QLearning(42L, 7);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> brain.load(file.toString()));
        assertTrue(e.getMessage().contains("720"), "어떤 차원이 어긋났는지 알려줘야 한다: " + e.getMessage());

        // 거부된 뒤 테이블이 반쯤 덮어써지지 않았는지 (헤더 검증이 읽기보다 먼저여야 한다)
        for (double v : brain.qCommon[0]) {
            assertEquals(0.0, v, 0.0, "거부된 로드가 테이블을 건드렸다");
        }
    }

    @Test
    @DisplayName("행동 수는 1~7만 허용한다")
    void 행동수_범위() {
        assertThrows(IllegalArgumentException.class, () -> new QLearning(42L, 0));
        assertThrows(IllegalArgumentException.class, () -> new QLearning(42L, 8));
        assertThrows(IllegalArgumentException.class, () -> new QLearning(42L, -1));
    }
}
