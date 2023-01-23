#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 50005;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[100005];
void add(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
}

int n, m;
int dp[N];
int dfn[N], low[N], dfs_clock, stk[N], top, scc, sccno[N], siz[N], in[N], topo[N];
int G[105][105];
bool in_stk[N];

void tarjan(int u) {
    in_stk[u] = 1;
    stk[++top] = u;
    dfn[u] = low[u] = ++dfs_clock;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if (!dfn[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        } else if (in_stk[v]) low[u] = min(low[u], dfn[v]);
    }
    if (dfn[u] == low[u]) {
        int x;
        scc++;
        while (true) {
            x = stk[top];
            top--;
            sccno[x] = scc;
            in_stk[x] = 0;
            siz[scc]++;
            if (x == u) break;
        }
    }
}
void clear(vector<int> &v) {
    vector<int> vv;
    swap(v, vv);
}

void rmain() {
    scc = dfs_clock = top = 0;
    memset(G, 0x3f, sizeof G);
    memset(head, 0, sizeof head);
    ecnt = 0;
    for (int i = 0; i <= n; i++) {
        sccno[i] = in_stk[i] = in[i] = dfn[i] = low[i] = 0;
        dp[i] = 0x3f3f3f3f;
    }
    for (int i = 1; i <= m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        add(u, v, w);
    }
    for (int i = 0; i < n; i++) if (!dfn[i]) tarjan(i);
    for (int i = 0; i < n; i++) {
        for (int x = head[i]; x; x = e[x].nxt) {
            int j = e[x].v, w = e[x].w;
            int u = sccno[i], v = sccno[j];
            if (u == v) continue;
            dp[v] = min(dp[v], w);       
        }
    }
    dp[sccno[0]] = 0;
    ll ans = 0;
    for (int i = 1; i <= scc; i++) ans += dp[i];
    printf("%lld\n", ans);
}

int main() {
    while (~scanf("%d %d", &n, &m)) {
        rmain();
    }
    return 0;
}