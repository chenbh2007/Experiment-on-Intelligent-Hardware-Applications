#include "SVM.h"
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
static float StateVector[10][10]={{1,0,0,0,0,0,0,0,0,0},
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
Model *SVM[3];
Vec *State;
int Size=0,Dim=5,rand_rate=100,Step=100;
float decay_factor=0.99;
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
// ==============================================================
int SVM_predict_move(){
    float Weight[3];
    int i;
    for (i=0;i<3;i++)
        Weight[i]=Predict(SVM[i],State);
    if (Weight[0]>=Weight[1]&&Weight[0]>=Weight[2]) return 1;
    if (Weight[1]>=Weight[0]&&Weight[1]>=Weight[2]) return 2;
    return 0;
}
void send_move(int sock_fd,int round) {
    const char *moves[3] = {"rock", "paper", "scissors"};
    int choice;
    if (rand()%1000<rand_rate*(1-round/2000))choice=rand()%3;
        else choice=SVM_predict_move();
    //int choice=(round<=Size)?(rand()%3):(SVM_predict_move());
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
Vec *create_vec_clone(const Vec *src) {
    Vec *v = (Vec *)malloc(sizeof(Vec));
    v->Dim = src->Dim;
    v->X = (float *)malloc(src->Dim * sizeof(float));
    memcpy(v->X, src->X, src->Dim * sizeof(float));
    return v;
}

void SlideWindow(Vec *NewVec,int y,float decay,int step){
    int n;
    int i,j;
    unsigned char tmpy;
    float tmpa;
    for (i=0;i<3;i++){
        n=SVM[i]->Data->Num;
        for (j=0;j<n;j++)
            SVM[i]->Alpha[j]=SVM[i]->Alpha[j]*decay;
        float s=0;
        tmpy=(y==i)?1:-1;
        tmpa=SVM[i]->Alpha[0];
        if (tmpy==SVM[i]->Data->Y[0]){
            memmove(SVM[i]->Data->Y,SVM[i]->Data->Y+1,(n-1)*sizeof(signed char));
            memmove(SVM[i]->Alpha,SVM[i]->Alpha+1,(n-1)*sizeof(float));
            SVM[i]->Alpha[n-1]=tmpa;
            SVM[i]->Data->Y[n-1]=tmpy;
        }
        else{
            for (j=1;j<n;j++)
                if (SVM[i]->Data->Y[j]!=SVM[i]->Data->Y[0]&&SVM[i]->Alpha[j]>Eps)
                    s+=SVM[i]->Alpha[j];
            float rate=(s-tmpa)/s;
            for (j=1;j<n;j++)
                if (SVM[i]->Data->Y[j]!=SVM[i]->Data->Y[0]&&SVM[i]->Alpha[j]>Eps)
                    SVM[i]->Alpha[j]=rate*SVM[i]->Alpha[j];
            memmove(SVM[i]->Data->Y,SVM[i]->Data->Y+1,(n-1)*sizeof(signed char));
            memmove(SVM[i]->Alpha,SVM[i]->Alpha+1,(n-1)*sizeof(float));
            SVM[i]->Alpha[n-1]=0;
            SVM[i]->Data->Y[n-1]=tmpy;
        }
        free(SVM[i]->Data->Data[0]->X);
        free(SVM[i]->Data->Data[0]);
        memmove(SVM[i]->Data->Data,SVM[i]->Data->Data+1,(n-1)*sizeof(Vec*));
        SVM[i]->Data->Data[n-1]=create_vec_clone(NewVec);
        memset(SVM[i]->IS_VALID_E,0,n*sizeof(unsigned char));
        if (SVM[i]->K_ij){
            for (j=0;j<n;j++){
                if (SVM[i]->K_ij[j]) free(SVM[i]->K_ij[j]);
            }
            free(SVM[i]->K_ij);
            SVM[i]->K_ij=NULL;
        }
        TrainOnline(SVM[i],step);
    }
}

int main() {
    signal(SIGINT, handle_sigint);
    srand((unsigned)time(NULL));
    State=(Vec *)malloc(sizeof(Vec));
    int y;
    State->Dim=Dim*10;
    State->X=(float *)malloc(10*Dim*sizeof(float));
    int i;
    for (i=0;i<Dim;i++){
        memcpy(State->X+i*10,StateVector[9],10*sizeof(float));
    }
    char S_name[20],D_name[20];
    for (i=0;i<3;i++){
        SVM[i]=(Model *)calloc(1,sizeof(Model));
        snprintf(S_name,sizeof(S_name),"baselineRPS%d.pth",i);
        snprintf(D_name,sizeof(D_name),"DataBase%d.db",i);
        LoadState(SVM[i],S_name,D_name);
    }


    int sock_fd = connect_to_server(SERVER_HOST, SERVER_PORT);
    if (sock_fd < 0) return -1;

    // 发送 join 报文（打内置 AI 用口令 "house"）
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "type", "join");
    cJSON_AddStringToObject(req, "password", "house");
    cJSON_AddStringToObject(req, "name", "SVM_Bot");
    char *json_str = cJSON_PrintUnformatted(req);

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
        int cnt=0;

        while ((newline_pos = strchr(line_start, '\n')) != NULL) {
            *newline_pos = '\0';

            if (strlen(line_start) > 0) {
                cJSON *res = cJSON_Parse(line_start);
                if (res) {
                    cJSON *type_item = cJSON_GetObjectItemCaseSensitive(res, "type");
                    if (cJSON_IsString(type_item) && type_item->valuestring) {
                        const char *type = type_item->valuestring;

                        if (strcmp(type, "round") == 0) {
                            send_move(sock_fd,cnt);
                        } else if (strcmp(type, "result") == 0) {
                            cJSON *you = cJSON_GetObjectItemCaseSensitive(res, "you");
                            cJSON *opp = cJSON_GetObjectItemCaseSensitive(res, "opp");
                            cJSON *score = cJSON_GetObjectItemCaseSensitive(res, "score");
                            cJSON *round = cJSON_GetObjectItemCaseSensitive(res, "round");

                            if (you && opp && score && round) {
                                int y_m = move_to_id(you->valuestring[0]);
                                int o_m = move_to_id(opp->valuestring[0]);
                                memmove(State->X,State->X+10,10*(Dim-1)*sizeof(float));
                                memcpy(State->X+10*(Dim-1),StateVector[y_m*3+o_m],10*sizeof(float));
                                y=o_m;
                                cnt=round->valueint+1;
                                SlideWindow(State,y,decay_factor,Step);

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
    free(State);
    for (i=0;i<3;i++)
        free(SVM[i]);
    printf("客户端已正常退出。\n");
    return 0;
}
