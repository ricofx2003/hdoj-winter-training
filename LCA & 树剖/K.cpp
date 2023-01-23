#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 100005;

int n, m;
ll dp[N];
ll sum[N];
vector<int> G[N], L[N];
int top[N], dep[N], faz[N], siz[N], son[N];
int dfc, rnk[N], dfn[N];
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs1(v, u);
        siz[u] += siz[v];
        if (son[u] == -1 || siz[son[u]] < siz[v]) son[u] = v;
    }
}
void dfs2(int u, int fa, int tp) {
    dfn[u] = ++dfc;
    rnk[dfc] = u;
    top[u] = tp;
    if (son[u] != -1) dfs2(son[u], u, tp);
    for (int v : G[u]) {
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
int LCA(int u, int v) {
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]])
            u = faz[top[u]];
        else
            v = faz[top[v]];
    }
    return dep[u] > dep[v] ? v : u;
}
struct Chain {
    int u, v, w;
} C[N];
ll tr[N];
void add(int x, ll d) {
    for (; x <= n; x += x & -x) tr[x] += d;
}
ll qry(int x) {
    ll res = 0;
    for (; x > 0; x -= x & -x) res += tr[x];
    return res;
}
ll qry(int l, int r) {
    return qry(r) - qry(l - 1);
}
ll query(int u, int v) {
    ll re = 0;
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]]) {
            re += qry(dfn[top[u]], dfn[u]);
            u = faz[top[u]];
        } else {
            re += qry(dfn[top[v]], dfn[v]);
            v = faz[top[v]];
        }
    }
    if (dep[u] < dep[v]) re += qry(dfn[u], dfn[v]);
    else re += qry(dfn[v], dfn[u]);
    return re;
}
void dfs(int u, int fa) {
    sum[u] = dp[u] = 0;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        sum[u] += dp[v];
    }
    dp[u] = sum[u];
    ll re = 0;
    for (int no : L[u]) {
        dp[u] = max(dp[u], query(C[no].u, C[no].v) + C[no].w + sum[u]);
    }
    add(dfn[u], sum[u] - dp[u]);
}
void rmain() {
    dfc = 0;
    memset(tr, 0, sizeof tr);
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) G[i].clear(), L[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    for (int i = 1; i <= m; i++) {
        scanf("%d %d %d", &C[i].u, &C[i].v, &C[i].w);
        L[LCA(C[i].u, C[i].v)].push_back(i);
    }
    dfs(1, 1);
    printf("%lld\n", dp[1]);
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}