#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1005;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}
int n, m;
int dp[N];
void dfs(int u, int fa, const int lim) {
    dp[u] = -1;
    int re = 0;
    bool flag = 1, child = 0;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        child = 1;
        dfs(v, u, lim);
        if (w > lim) {
            if (dp[v] == -1) flag = 0;
            else re += dp[v];
        } else {
            if (dp[v] == -1) re += w;
            else re += min(w, dp[v]);
        }
    }
    if (child && flag) dp[u] = re;
}
bool check(int mid) {
    dfs(1, 1, mid);
    return dp[1] != -1 && dp[1] <= m;
}

void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    int l = 1, r = 0, ans = -1;
    for (int i = 1; i < n; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        add_edge(u, v, w);
        r = max(r, w);
    }
    while (l <= r) {
        int mid = (l + r ) >> 1;
        if (check(mid)) r = mid - 1, ans = mid;
        else l = mid + 1;
    }
    printf("%d\n", ans);
}
int main() {
    while (~scanf("%d %d", &n, &m) && n) {
        rmain();
    }
    return 0;
}