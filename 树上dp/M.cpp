#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 105;

int dp[N][N], cost[N], n, m, gold[N];
vector<int> G[N];

void dfs(int u, int fa) {
    if (cost[u] <= m) fill(dp[u] + cost[u], dp[u] + m + 1, gold[u]);
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        for (int i = m; i >= cost[u]; i--) {
            for (int j = max(cost[v], 1); j <= i - cost[u]; j++) {
                dp[u][i] = max(dp[u][i], dp[v][j] + dp[u][i - j]);
            }
        }
    }
}

void rmain() {
    memset(dp, 0, sizeof dp);
    for (int i = 1; i <= n; i++) {
        int x;
        G[i].clear();
        scanf("%d %d", &x, &gold[i]);
        cost[i] = (x + 19) / 20;
    }
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    dfs(1, 1);
    if (m == 0) puts("0");
    else printf("%d\n", dp[1][m]);
}
int main() {
    while (~scanf("%d %d", &n, &m) && n != -1) {
        rmain();
    }
    return 0;
}