package com.mario.rl.network;

import com.google.gson.Gson;
import com.mario.rl.model.Action;
import com.mario.rl.model.GameState;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * Python 게임 서버와 TCP 소켓으로 통신하는 클라이언트.
 *
 * <p>Python(서버)이 먼저 떠 있어야 하므로, 연결이 거부되면 일정 간격으로 재시도한다.
 * 메시지는 한 줄(개행 {@code \n} 구분) 단위의 UTF-8 JSON으로 주고받는다.</p>
 *
 * <ul>
 *   <li>{@link #receiveGameState()} — Python이 보낸 상태 JSON 한 줄을 {@link GameState}로 역직렬화</li>
 *   <li>{@link #sendAction(Action)} — 선택한 행동을 {@code {"action": N}} JSON으로 직렬화하여 전송</li>
 * </ul>
 *
 * <p>{@link Closeable}을 구현하므로 try-with-resources로 소켓을 안전하게 닫을 수 있다.</p>
 */
public class SocketClient implements Closeable {

    /** 연결 재시도 최대 횟수. */
    private static final int MAX_RETRIES = 10;
    /** 연결 재시도 간격(밀리초). */
    private static final long RETRY_DELAY_MS = 3000;

    private final Gson gson = new Gson();
    private final Socket socket;
    private final BufferedReader reader;
    private final BufferedWriter writer;

    /**
     * 지정한 호스트/포트의 Python 서버에 연결한다. 연결이 거부되면 {@value #RETRY_DELAY_MS}ms 간격으로
     * 최대 {@value #MAX_RETRIES}회 재시도한다.
     *
     * @param host 서버 호스트 (예: "localhost")
     * @param port 서버 포트 (예: 9999)
     * @throws IOException          최대 재시도 후에도 연결에 실패한 경우
     * @throws InterruptedException 재시도 대기 중 인터럽트된 경우
     */
    public SocketClient(String host, int port) throws IOException, InterruptedException {
        this.socket = connectWithRetry(host, port);
        this.reader = new BufferedReader(
                new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
        this.writer = new BufferedWriter(
                new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
    }

    /** 연결에 성공할 때까지(또는 최대 횟수까지) 재시도하며 연결된 소켓을 반환한다. */
    private Socket connectWithRetry(String host, int port)
            throws IOException, InterruptedException {
        IOException lastError = null;
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                Socket s = new Socket();
                s.connect(new InetSocketAddress(host, port));
                System.out.printf("[SocketClient] 연결 성공: %s:%d%n", host, port);
                return s;
            } catch (IOException e) {
                lastError = e;
                System.out.printf("[SocketClient] 연결 실패 (%d/%d). %dms 후 재시도...%n",
                        attempt, MAX_RETRIES, RETRY_DELAY_MS);
                Thread.sleep(RETRY_DELAY_MS);
            }
        }
        throw new IOException(
                "Python 서버 연결 실패: " + host + ":" + port + " (" + MAX_RETRIES + "회 시도)",
                lastError);
    }

    /**
     * Python으로부터 게임 상태 JSON 한 줄을 수신하여 {@link GameState}로 변환한다.
     *
     * @return 수신한 게임 상태
     * @throws IOException 연결이 끊겼거나 읽기에 실패한 경우
     */
    public GameState receiveGameState() throws IOException {
        String line = reader.readLine();
        if (line == null) {
            throw new IOException("서버 연결이 종료되었습니다 (수신 데이터 없음).");
        }
        return gson.fromJson(line, GameState.class);
    }

    /**
     * 선택한 행동을 {@code {"action": N}} 형태의 JSON으로 직렬화하여 Python에 전송한다.
     * 메시지 끝에 개행을 붙이고 즉시 flush한다.
     *
     * @param action 전송할 행동
     * @throws IOException 쓰기에 실패한 경우
     */
    public void sendAction(Action action) throws IOException {
        String json = "{\"action\": " + action.getValue() + "}";
        writer.write(json);
        writer.write("\n");
        writer.flush();
    }

    /** 소켓과 입출력 스트림을 닫는다. */
    @Override
    public void close() throws IOException {
        socket.close();
    }
}
