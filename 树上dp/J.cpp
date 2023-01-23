#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 105;

int n, T, f[N][505], a[N];
int head[N], ecnt;
int stk[N], top, qwq[N], dp[505];

struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}

void dfs1(int u, int fa) {
    fill(f[u], f[u] + T + 1, a[u]);
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        dfs1(v, u);
        if (stk[top] == v) {
            stk[++top] = u;
            qwq[top] = w;
            continue;
        }
        for (int i = T; i >= 0; i--) {
            for (int j = w * 2; j <= i; j++) {
                f[u][i] = max(f[u][i], f[v][j - w * 2] + f[u][i - j]);
            }
        }
    }
}

void rmain() {
    memset(dp, 0, sizeof dp);
    memset(head, 0, sizeof head);
    ecnt = 0;
    top = 0;
    for (int i = 2; i <= n; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        add_edge(u, v, w);
    }
    for (int i = 1; i <= n; i++) {
        scanf("%d", &a[i]);
    }
    stk[++top] = n;
    qwq[top] = 0;
    dfs1(1, 1);
    int qaq = 0;
    int cur = 0;
    for (int k = top; k >= 1; k--) {
        int u = stk[k];
        cur ^= 1;
        if (k == top) {
            memcpy(dp, f[u], sizeof dp);
            qaq += qwq[k];
            continue;
        }
        if (qaq > T) {
            puts("Human beings die in pursuit of wealth, and birds die in pursuit of food!");
            return;
        }
        for (int i = T; i >= qaq; i--) {
            dp[i] = -1;
            for (int j = 0; j <= i - qaq; j++) {
                if (i - qwq[k + 1] - j >= qaq - qwq[k + 1]) dp[i] = max(dp[i], f[u][j] + dp[i - qwq[k + 1] - j]);
            }
        }
        qaq += qwq[k];
    }
    if (qaq > T) puts("Human beings die in pursuit of wealth, and birds die in pursuit of food!");
    else printf("%d\n", dp[T]);
}
int main() {
    while (~scanf("%d %d", &n, &T)) {
        rmain();
    }
    return 0;
}