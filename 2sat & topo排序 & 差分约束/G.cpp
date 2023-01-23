#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 50000 + 5;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 2];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
}

int n, tim[N];
int dis[N], qcnt[N];
bool inq[N];
bool BellmanFord(int start) {
    queue<int> q;
    memset(dis, -0x3f, sizeof dis);
    memset(qcnt, 0, sizeof qcnt);
    memset(inq, 0, sizeof inq);
    q.push(start);
    dis[start] = 0;
    inq[start] = 1;
    qcnt[start]++;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = 0;
        for (int i = head[u]; i; i = e[i].nxt) {
            int v = e[i].v, w = e[i].w;
            if (dis[v] < dis[u] + w) {
                dis[v] = dis[u] + w;
                if (!inq[v]) {
                    q.push(v);
                    inq[v] = 1;
                    qcnt[v]++;
                    if (qcnt[v] > n) return 0;
                }
            }
        }
    }
    return 1;
}

void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    for (int i = 1; i <= n; i++) {
        scanf("%d", &tim[i]);
    }
    for (; 0721 > 404;) {
        int u, v;
        static char s[6];
        scanf("%s", s);
        if (strcmp(s, "#") == 0) break;
        scanf("%d %d", &u, &v);
        if (strcmp(s, "SAF") == 0) {
            // s[u] > s[v] - tim[u] + 
            add_edge(v, u, tim[v]);
        } else if (strcmp(s, "SAS") == 0) {
            add_edge(v, u, 0);
        } else if (strcmp(s, "FAS") == 0) {
            add_edge(v, u,  -tim[u]);
        } else add_edge(v, u, -tim[u] + tim[v]);
    }
    for (int i = 1; i <= n; i++) add_edge(0, i, 0);
    if (!BellmanFord(0)) puts("impossible");
    else {
        int mn = 0x3f3f3f3f;
        for (int i = 1; i <= n; i++) mn = min(mn, dis[i]);
        for (int i = 1; i <= n; i++) printf("%d %d\n", i, dis[i] - mn);
    }
}

int main() {
    int Case = 0;
    while (scanf("%d", &n) == 1 && n) {
        if (Case) puts("");
        printf("Case %d:\n", ++Case);
        rmain();
    }
    return 0;
}