#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1505;
int n;
int dp[N][2];
vector<int> G[N];

void dfs(int u, int fa) {
    dp[u][1] = 1;
    dp[u][0] = 0;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        dp[u][0] += dp[v][1];
        dp[u][1] += min(dp[v][1], dp[v][0]);
    }
}

void rmain() {
    for (int i = 0; i < n; i++) G[i].clear();
    for (int i = 0; i < n; i++) {
        int u, num, v;
        scanf("%d:(%d)", &u, &num);
        for (int j = 1; j <= num; j++) {
            scanf("%d", &v);
            G[u].push_back(v);
            G[v].push_back(u);
        }
    }
    dfs(1, 1);
    printf("%d\n", min(dp[1][0] ,dp[1][1]));
}
int main() {
    while (~scanf("%d", &n)) {
        rmain();
    }
    return 0;
}