#include "SVM.h"
#include <cjson/cJSON.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <math.h>

#define SERVER_HOST "eelab.madeinpku.club"
#define SERVER_PORT "8888"
#define BUFFER_SIZE 8192
static volatile sig_atomic_t stop_requested = 0;
const unsigned char StateVector[10][10]={{1,0,0,0,0,0,0,0,0,0},
                                {0,1,0,0,0,0,0,0,0,0},
                                {0,0,1,0,0,0,0,0,0,0},
                                {0,0,0,1,0,0,0,0,0,0},
                                {0,0,0,0,1,0,0,0,0,0},
                                {0,0,0,0,0,1,0,0,0,0},
                                {0,0,0,0,0,0,1,0,0,0},
                                {0,0,0,0,0,0,0,1,0,0},
                                {0,0,0,0,0,0,0,0,1,0},
                                {0,0,0,0,0,0,0,0,0,1}
};
void handle_sigint(int sig) {
    (void)sig;
    stop_requested = 1;
}

int connect_to_host(const char *hostname, const char *port_str) {
    struct addrinfo hints, *result = NULL, *rp = NULL;
    int sock_fd = -1;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int status = getaddrinfo(hostname, port_str, &hints, &result);
    if (status != 0) {
        fprintf(stderr, "DNS 解析失败 [%s]: %s\n", hostname, gai_strerror(status));
        return -1;
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd == -1) continue;

        if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            printf("成功连接到服务器: %s\n", hostname);
            break;
        }
        close(sock_fd);
        sock_fd = -1;
    }
    freeaddrinfo(result);
    return sock_fd;
}
Model *SVM[3];
Vec *State;
int Dim,n;
unsigned char History[50];
// 统一出拳并发送
void send_move(int sock_fd) {
    const char *moves[3] = {"rock", "paper", "scissors"};
    State=(Vec *)malloc(sizeof(Vec));
    State->Dim=Dim;
    State->X=(float *)malloc(Dim*sizeof(float));
    float Weight[3];
    int i,j;
    for (i=0;i<n;i++){
        for (j=0;j<10;j++)
            State->X[i*10+j]=StateVector[History[i]][j];
    }
    for (i=0;i<3;i++){
        Weight[i]=Predict(SVM[i],State);
    }
    free(State->X);
    free(State);
    int choice;
    if (rand()%100<90) choice=rand()%3;
    else
    if (Weight[0]>=Weight[1]&&Weight[0]>=Weight[2]) choice=1;
    else 
        if (Weight[1]>=Weight[0]&&Weight[1]>=Weight[2]) choice=2;
            else choice=0;

    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "type", "move");
    cJSON_AddStringToObject(req, "move", moves[choice]);

    char *json_str = cJSON_PrintUnformatted(req);
    char send_buf[256];
    snprintf(send_buf, sizeof(send_buf), "%s\n", json_str);

    cJSON_Delete(req);
    free(json_str);
    
    send(sock_fd, send_buf, strlen(send_buf), 0);
    // printf("已出拳: %s", send_buf);
}
int main(void) {
    signal(SIGINT, handle_sigint);
    srand((unsigned)time(NULL));

    int sock_fd = connect_to_host(SERVER_HOST, SERVER_PORT);
    if (sock_fd < 0) return -1;

    // 1. 发送 join 报文
    cJSON *req_root = cJSON_CreateObject();
    cJSON_AddStringToObject(req_root, "type", "join");
    cJSON_AddStringToObject(req_root, "password", "house"); // 对战内置 AI
    cJSON_AddStringToObject(req_root, "name", "Alice");

    char *json_str = cJSON_PrintUnformatted(req_root);
    char send_buf[512];
    snprintf(send_buf, sizeof(send_buf), "%s\n", json_str);
    cJSON_Delete(req_root);
    free(json_str);

    //SVM模型初始化
    char state_file[20],data_set[20];
    int i;
    for (i=0;i<3;i++){
        snprintf(state_file,sizeof(state_file),"baselineRPS%d.pth",i);
        snprintf(data_set,sizeof(data_set),"DataBase%d.db",i);
        SVM[i]=(Model *)malloc(sizeof(Model));
        LoadState(SVM[i],state_file,data_set);
    }
    Dim=SVM[0]->Data->Data[0]->Dim;
    n=Dim/10;
    for (i=0;i<n;i++) History[i]=9;

    if (send(sock_fd, send_buf, strlen(send_buf), 0) < 0) {
        perror("发送 join 失败");
        close(sock_fd);
        return -1;
    }
    printf("已发送 Join 申请: %s", send_buf);

    FILE *f = fopen("DataSet.db", "a");
    if (!f) {
        perror("打开 DataSet.db 失败");
        close(sock_fd);
        for (i=0;i<3;i++) free(SVM[i]);
        free(State);
        return -1;
    }

    // 环形/行拼接缓冲区
    char stream_buf[BUFFER_SIZE];
    size_t stream_len = 0;
    int is_running = 1;

    while (is_running && !stop_requested) {
        // 从 TCP 流中读取数据
        ssize_t read_bytes = recv(sock_fd, stream_buf + stream_len, sizeof(stream_buf) - 1 - stream_len, 0);
        if (read_bytes <= 0) {
            if (read_bytes < 0 && (errno == EINTR || stop_requested)) break;
            printf("服务器断开连接\n");
            break;
        }

        stream_len += read_bytes;
        stream_buf[stream_len] = '\0';

        // 核心修复：按 '\n' 逐行切割并解析 JSON
        char *line_start = stream_buf;
        char *newline_pos = NULL;

        while ((newline_pos = strchr(line_start, '\n')) != NULL) {
            *newline_pos = '\0'; // 将换行符截断为字符串结束符

            if (strlen(line_start) > 0) {
                cJSON *res = cJSON_Parse(line_start);
                if (res) {
                    cJSON *type_item = cJSON_GetObjectItemCaseSensitive(res, "type");
                    if (cJSON_IsString(type_item) && type_item->valuestring) {
                        const char *type = type_item->valuestring;

                        if (strcmp(type, "round") == 0) {
                            // 收到 round，必须立刻出拳！
                            send_move(sock_fd);
                        } else if (strcmp(type, "result") == 0) {
                            // 记录本轮结果到数据集
                            cJSON *round = cJSON_GetObjectItemCaseSensitive(res, "round");
                            cJSON *you = cJSON_GetObjectItemCaseSensitive(res, "you");
                            cJSON *opp = cJSON_GetObjectItemCaseSensitive(res, "opp");
                            cJSON *score = cJSON_GetObjectItemCaseSensitive(res, "score");
                            int result=(you->valuestring[0]=='r'?0:(you->valuestring[0]=='p'?1:2))*3+(opp->valuestring[0]=='r'?0:(opp->valuestring[0]=='p'?1:2));
                            for (i=0;i<n-1;i++){
                                History[i]=History[i+1];
                            }
                            History[n-1]=result;

                            if (round && you && opp && score) {
                                fprintf(f, "%d ", round->valueint);
                                fprintf(f, "%d ", you->valuestring[0] == 'r' ? 0 : (you->valuestring[0] == 'p' ? 1 : 2));
                                fprintf(f, "%d ", opp->valuestring[0] == 'r' ? 0 : (opp->valuestring[0] == 'p' ? 1 : 2));

                                cJSON *s_you = cJSON_GetObjectItemCaseSensitive(score, "you");
                                cJSON *s_opp = cJSON_GetObjectItemCaseSensitive(score, "opp");
                                cJSON *s_draw = cJSON_GetObjectItemCaseSensitive(score, "draw");
                                if (s_you && s_opp && s_draw) {
                                    fprintf(f, "%d %d %d\n", s_you->valueint, s_opp->valueint, s_draw->valueint);
                                    fflush(f);
                                }
                            }
                        } else if (strcmp(type, "matched") == 0) {
                            printf("匹配成功，对局开始！\n");
                        } else if (strcmp(type, "gameover") == 0) {
                            printf("对局正常结束 (gameover)\n");
                            is_running = 0;
                        } else if (strcmp(type, "opponent_left") == 0) {
                            printf("对手中途退出\n");
                            is_running = 0;
                        } else if (strcmp(type, "error") == 0) {
                            cJSON *msg = cJSON_GetObjectItemCaseSensitive(res, "msg");
                            if (msg) fprintf(stderr, "服务器告警: %s\n", msg->valuestring);
                        }
                    }
                    cJSON_Delete(res); // 安全释放单行 JSON
                }
            }
            line_start = newline_pos + 1; // 移向下一行
        }

        // 将未读完的残留半截数据（如果有）平移到缓冲区头部
        size_t processed = line_start - stream_buf;
        if (processed < stream_len) {
            memmove(stream_buf, line_start, stream_len - processed);
            stream_len -= processed;
        } else {
            stream_len = 0;
        }
    }

    if (f) fclose(f);
    if (sock_fd >= 0) close(sock_fd);
    for (i=0;i<3;i++) free(SVM[i]);
    printf("程序退出，数据集已保存。\n");
    return 0;
}
