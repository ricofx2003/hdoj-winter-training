#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 10005;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}
int n, m, root;
int dp[N][15];

void dfs(int u, int fa) {
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        dfs(v, u);
        for (int i = m; i >= 0; i--) {
            dp[u][i] += dp[v][0] + 2 * w;
            for (int j = 1; j <= i; j++) {
                dp[u][i] = min(dp[u][i], dp[u][i - j] + dp[v][j] + j * w);
            }
        }
    }
}

void rmain() {
    memset(dp, 0, sizeof dp);
    ecnt = 0;
    memset(head, 0, sizeof head);
    for(int i = 1; i < n; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        add_edge(u, v, w);
    }
    dfs(root, root);
    printf("%d\n", dp[root][m]);
}
int main() {
    while (~scanf("%d %d %d", &n, &root, &m)) {
        rmain();
    }
    return 0;
}