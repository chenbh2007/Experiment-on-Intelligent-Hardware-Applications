#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <cjson/cJSON.h>

#define SERVER_HOST "eelab.madeinpku.club"
#define SERVER_PORT "8888"
#define BUFFER_SIZE 8192

static volatile sig_atomic_t g_stop = 0;

void handle_sigint(int sig) {
    (void)sig;
    g_stop = 1;
}

// 建立 TCP 连接
int connect_to_server(const char *host, const char *port) {
    struct addrinfo hints, *res, *rp;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port, &hints, &res) != 0) {
        perror("DNS 解析失败");
        return -1;
    }

    int sock_fd = -1;
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd < 0) continue;
        if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            printf("成功连接至对战服务器: %s:%s\n", host, port);
            break;
        }
        close(sock_fd);
        sock_fd = -1;
    }
    freeaddrinfo(res);
    return sock_fd;
}

// 手势映射转换
static inline int move_to_id(char c) {
    if (c == 'r') return 0; // rock
    if (c == 'p') return 1; // paper
    return 2;               // scissors
}

// ==================== 马尔可夫在线策略核心 ====================
static int g_trans_count[9][3] = {0}; // 9种状态 -> 对手3种动作的转移频数
static int g_last_state = -1;         // 上一局双方状态 (-1 表示第 1 轮)

// 决策出拳
int markov_predict_move(void) {
    // 1. 冷启动（第 1 轮）或 10% 探索扰动：随机出拳
    if (g_last_state < 0 || (rand() % 100) < 10) {
        return rand() % 3;
    }

    // 2. 统计对手在上一轮状态下各出拳的历史频数
    int r_cnt = g_trans_count[g_last_state][0];
    int p_cnt = g_trans_count[g_last_state][1];
    int s_cnt = g_trans_count[g_last_state][2];
    int total = r_cnt + p_cnt + s_cnt;

    // 若当前状态此前从未遇到过，随机出拳
    if (total == 0) {
        return rand() % 3;
    }

    // 3. 找出对手最可能出的手势
    int pred_opp = 0;
    if (p_cnt > r_cnt && p_cnt >= s_cnt) pred_opp = 1;
    else if (s_cnt > r_cnt && s_cnt > p_cnt) pred_opp = 2;

    // 4. 克制对手：出石头克剪刀(0克2)，出布克石头(1克0)，出剪刀克布(2克1)
    return (pred_opp + 1) % 3;
}

// 在线学习：根据当轮真实结果更新频数表
void markov_update_model(int you_move, int opp_move) {
    // 若存在上一轮状态，则记录该转移
    if (g_last_state >= 0) {
        for (int a = 0; a < 3; a++) {
            g_trans_count[g_last_state][a] =(int)(g_trans_count[g_last_state][a] * 0.95);
        }
        g_trans_count[g_last_state][opp_move]++;
    }
    // 更新当前轮状态供下一轮决策使用 (you * 3 + opp)
    g_last_state = you_move * 3 + opp_move;
}
// ==============================================================

void send_move(int sock_fd) {
    const char *moves[3] = {"rock", "paper", "scissors"};
    int choice = markov_predict_move();

    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "type", "move");
    cJSON_AddStringToObject(req, "move", moves[choice]);
    char *json_str = cJSON_PrintUnformatted(req);

    char send_buf[256];
    snprintf(send_buf, sizeof(send_buf), "%s\n", json_str);
    cJSON_Delete(req);
    free(json_str);

    send(sock_fd, send_buf, strlen(send_buf), 0);
}

int main(void) {
    signal(SIGINT, handle_sigint);
    srand((unsigned)time(NULL));

    int sock_fd = connect_to_server(SERVER_HOST, SERVER_PORT);
    if (sock_fd < 0) return -1;

    // 发送 join 报文（打内置 AI 用口令 "house"）
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "type", "join");
    cJSON_AddStringToObject(req, "password", "house");
    cJSON_AddStringToObject(req, "name", "MarkovBot");
    char *json_str = cJSON_PrintUnformatted(req);
    FILE *f=fopen("DataSet.db","a");

    char send_buf[512];
    snprintf(send_buf, sizeof(send_buf), "%s\n", json_str);
    cJSON_Delete(req);
    free(json_str);

    if (send(sock_fd, send_buf, strlen(send_buf), 0) < 0) {
        perror("发送 join 失败");
        close(sock_fd);
        return -1;
    }

    char stream_buf[BUFFER_SIZE];
    size_t stream_len = 0;
    int is_running = 1;

    while (is_running && !g_stop) {
        ssize_t read_bytes = recv(sock_fd, stream_buf + stream_len, sizeof(stream_buf) - 1 - stream_len, 0);
        if (read_bytes <= 0) {
            if (read_bytes < 0 && (errno == EINTR || g_stop)) break;
            printf("服务器断开连接\n");
            break;
        }
        stream_len += read_bytes;
        stream_buf[stream_len] = '\0';

        // 行级拆包解析
        char *line_start = stream_buf;
        char *newline_pos = NULL;

        while ((newline_pos = strchr(line_start, '\n')) != NULL) {
            *newline_pos = '\0';

            if (strlen(line_start) > 0) {
                cJSON *res = cJSON_Parse(line_start);
                if (res) {
                    cJSON *type_item = cJSON_GetObjectItemCaseSensitive(res, "type");
                    if (cJSON_IsString(type_item) && type_item->valuestring) {
                        const char *type = type_item->valuestring;

                        if (strcmp(type, "round") == 0) {
                            send_move(sock_fd);
                        } else if (strcmp(type, "result") == 0) {
                            cJSON *you = cJSON_GetObjectItemCaseSensitive(res, "you");
                            cJSON *opp = cJSON_GetObjectItemCaseSensitive(res, "opp");
                            cJSON *score = cJSON_GetObjectItemCaseSensitive(res, "score");
                            cJSON *round = cJSON_GetObjectItemCaseSensitive(res, "round");

                            if (you && opp && score && round) {
                                int y_m = move_to_id(you->valuestring[0]);
                                int o_m = move_to_id(opp->valuestring[0]);
                                markov_update_model(y_m, o_m);
                                fprintf(f,"%d %d\n",round->valueint,y_m*3+o_m);

                                cJSON *s_you = cJSON_GetObjectItemCaseSensitive(score, "you");
                                cJSON *s_opp = cJSON_GetObjectItemCaseSensitive(score, "opp");
                                if (round->valueint % 50 == 0 && s_you && s_opp) {
                                    printf("第 %4d 轮 | 比分: %d 胜 - %d 负\n",
                                           round->valueint, s_you->valueint, s_opp->valueint);
                                }
                            }
                        } else if (strcmp(type, "matched") == 0) {
                            printf("匹配完成，对局启动！\n");
                        } else if (strcmp(type, "gameover") == 0) {
                            cJSON *score = cJSON_GetObjectItemCaseSensitive(res, "score");
                            cJSON *winner = cJSON_GetObjectItemCaseSensitive(res, "winner");
                            if (score && winner) {
                                printf("\n=== 游戏结束 (胜者: %s) ===\n", winner->valuestring);
                                printf("最终比分: 我方 %d 胜 | 对手 %d 胜 | 平局 %d\n",
                                       cJSON_GetObjectItem(score, "you")->valueint,
                                       cJSON_GetObjectItem(score, "opp")->valueint,
                                       cJSON_GetObjectItem(score, "draw")->valueint);
                            }
                            is_running = 0;
                        } else if (strcmp(type, "opponent_left") == 0) {
                            printf("对手离开对局\n");
                            is_running = 0;
                        } else if (strcmp(type, "error") == 0) {
                            cJSON *msg = cJSON_GetObjectItemCaseSensitive(res, "msg");
                            if (msg) fprintf(stderr, "服务器警告: %s\n", msg->valuestring);
                        }
                    }
                    cJSON_Delete(res);
                }
            }
            line_start = newline_pos + 1;
        }

        // 剩余未完整读取的包平移到头部
        size_t processed = line_start - stream_buf;
        if (processed < stream_len) {
            memmove(stream_buf, line_start, stream_len - processed);
            stream_len -= processed;
        } else {
            stream_len = 0;
        }
    }

    close(sock_fd);
    printf("客户端已正常退出。\n");
    fclose(f);
    return 0;
}
