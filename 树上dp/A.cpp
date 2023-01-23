#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 6005;

int n, dp[N][2], a[N];
vector<int> G[N];
bool vis[N];

void dfs(int u, int fa) {
    vis[u] = 1;
    dp[u][0] = 0;
    dp[u][1] = a[u];
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        dp[u][0] += max(dp[v][1], dp[v][0]);
        dp[u][1] += dp[v][0];
    }
}

void rmain() {
    for (int i = 1; i <= n; i++)scanf("%d", &a[i]), G[i].clear();
    memset(vis, 0, sizeof vis);
    while (true) {
        int u, v;
        scanf("%d %d", &u, &v);
        if (u == 0) break;
        G[u].push_back(v);
        G[v].push_back(u);
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) if (!vis[i]) {
        dfs(i, i);
        ans += max(dp[i][0], dp[i][1]);
    }
    printf("%d\n", ans);
}
int main() {
    while (~scanf("%d", &n) && n) {
        rmain();
    }
    return 0;
}