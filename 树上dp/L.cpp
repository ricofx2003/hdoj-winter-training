#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 205;

int n, m;
vector<int> G[N];
int dp[N][N], in[N], val[N];
bitset<205> f[N];

void dfs(int u) {
    fill(dp[u] + 1, dp[u] + m + 1, val[u]);
    for (int v : G[u]) {
        //if (f[v][v]) continue;
        dfs(v);
        for (int i = m; i; i--) {
            for (int j = 1; j < i; j++) {
                dp[u][i] = max(dp[u][i], dp[v][j] + dp[u][i - j]);
            }
        }
    }
}

void rmain() {
    m += 1;
    for (int i = 0; i <= n; i++) f[i].reset(), G[i].clear(), in[i] = 0;
    for (int i = 1; i <= n; i++) {
        int a;
        scanf("%d %d", &a, &val[i]);
        if (a == 0) continue;
        in[i]++;
        f[a].set(i);
        G[a].push_back(i);
    }
    for (int i = 1; i <= n; i++) if (!in[i]) G[0].push_back(i), f[0].set(i);
    for (int k = 0; k <= n; k++) {
        for (int i = 0; i <= n; i++) if (f[i][k]) f[i] |= f[k];
    }
    dfs(0);
    printf("%d\n", dp[0][m]);
}
int main() {
    while (~scanf("%d %d", &n, &m) && (n + m)) {
        rmain();
    }
    return 0;
}