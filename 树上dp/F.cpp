#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 10007;
const int N = 205;

int n;
vector<int> G[N];

void add(int &a, int b) {
    ((a += b) >= mod) && (a -= mod);
}

int cnt[2], siz[N], MAX[N];
void calc(int u, int fa) {
    siz[u] = 1;
    MAX[u] = 0;
    for (int v : G[u]) {
        if (v == fa) continue;
        calc(v, u);
        siz[u] += siz[v];
        MAX[u] = max(MAX[u], siz[v]);
    }
    MAX[u] = max(MAX[u], n - siz[u]);
    if (MAX[u] <= n / 2) cnt[cnt[0] != 0] = u;
}

int dp[N][N], ex[N];
void dfs(int u, int fa, int lim = 0) {
    dp[u][1] = 1;
    siz[u] = 1;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u, lim);
        if (lim == v) continue;
        for (int i = siz[u] + siz[v]; i > 1; i--) {
            for (int j = max(1, i - siz[u]); j <= siz[v] && j < i; j++) {
                add(dp[u][i], dp[u][i - j] * 1ll * dp[v][j] % mod);
            }
        }
        siz[u] += siz[v];
    }
}
/* 草，读错题了
void rmain() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) G[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    cnt[0] = cnt[1] = 0;
    calc(1, 1);
    if (cnt[1] == 0) {
        int ans = 0;
        for (int u = 1; u <= n; u++) {
            memset(dp, 0, sizeof dp);
            dfs(u, u);
            for (int v : G[u]) {
                memset(ex, 0, sizeof ex);
                ex[1] = 1;
                int ez = 1;
                for (int ev : G[u]) {
                    if (ev == v) continue;
                    for (int i = ez + siz[ev]; i > 1; i--) {
                        for (int j = max(1, i - ez); j < i && j <= siz[ev]; j++) {
                            add(ex[i], ex[i - j] * 1ll * dp[ev][j] % mod);
                        }
                    }
                }
                int Ex = 0;
                for (int i = 1; i <= n; i++) {
                    for (int j = i / 2 + 1; j < i; j++) {
                        add(Ex, dp[v][j] * 1ll * ex[i - j] % mod);
                    }
                }
                add(ans, mod - Ex);
            }
            for (int i = 1; i <= n; i++) add(ans, dp[u][i]);
        }
        printf("%d\n", ans);
    } else puts("-1");
}*/
void real_main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) G[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    cnt[0] = cnt[1] = 0;
    calc(1, 1);
    if (cnt[1] == 0) {
        int ans = 0;
        for (int u = cnt[0]; u <= cnt[0]; u++) {
            memset(dp, 0, sizeof dp);
            dfs(u, u);
            for (int v : G[u]) {
                memset(ex, 0, sizeof ex);
                ex[1] = 1;
                int ez = 1;
                for (int ev : G[u]) {
                    if (ev == v) continue;
                    for (int i = ez + siz[ev]; i > 1; i--) {
                        for (int j = max(1, i - ez); j < i && j <= siz[ev]; j++) {
                            add(ex[i], ex[i - j] * 1ll * dp[ev][j] % mod);
                        }
                    }
                    ez += siz[ev];
                }
                int Ex = 0;
                for (int i = 1; i <= n; i++) {
                    for (int j = (i - 1) / 2 + 1; j <= siz[v] && j < i; j++) {
                        add(Ex, dp[v][j] * 1ll * ex[i - j] % mod);
                    }
                }
                add(ans, mod - Ex);
            }
            for (int i = 1; i <= n; i++) add(ans, dp[u][i]);
        }
        printf("%d\n", ans);
    } else {
        int ans = 0;
        dfs(cnt[0], cnt[0], cnt[1]);
        for (int i = 1; i <= n / 2; i++) add(ans, dp[cnt[0]][i] * 1ll * dp[cnt[1]][i] % mod);
        printf("%d\n", ans);
    }
}

int main() {
    int T, qwq = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case %d: ", ++qwq);
        real_main();
    }
    return 0;
}