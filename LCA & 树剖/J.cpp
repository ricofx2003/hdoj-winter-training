#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 1e5 + 5;

vector<int> G[N];
int n, top[N], dep[N], faz[N], siz[N], son[N];
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
#define mid ((l + r) >> 1)
#define lson (x << 1)
#define rson ((x << 1) | 1) 
int sum[N << 2], tag[N << 2], pre[N], val[N];
void pushup(int x) {
    sum[x] = sum[lson] + sum[rson];
}
void build(int x, int l, int r) {
    tag[x] = -1;
    if (l == r) {
        sum[x] = 0;
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    pushup(x);
}
void add(int x, int l, int r, int d) {
    if (d == 1) sum[x] = pre[r] - pre[l - 1];
    else sum[x] = 0;
    tag[x] = d;
}
void pushdown(int x, int l, int r) {
    if (tag[x] != -1) {
        add(lson, l, mid, tag[x]);
        add(rson, mid + 1, r, tag[x]);
        tag[x] = -1;
    }
}
void modify(int x, int l, int r, int L, int R, int d) {
    if (l > R || r < L) return;
    if (l >= L && r <= R) {
        add(x, l, r, d);
        return;
    }
    pushdown(x, l, r);
    modify(lson, l, mid, L, R, d);
    modify(rson, mid + 1, r, L, R, d);
    pushup(x);
}
void occupy(int u, int v) {
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]]) {
            modify(1, 1, n, dfn[top[u]], dfn[u], 1);
            u = faz[top[u]];
        } else {
            modify(1, 1, n, dfn[top[v]], dfn[v], 1);
            v = faz[top[v]];
        }
    }
    if (dep[u] < dep[v]) modify(1, 1, n, dfn[u], dfn[v], 1);
    else modify(1, 1, n, dfn[v], dfn[u], 1);
}

void rmain() {
    dfc = 0;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &val[i]), G[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    for (int l = 1; l <= n; l++) pre[l] = pre[l - 1] + val[rnk[l]];
    build(1, 1, n);
    int Q;
    scanf("%d", &Q);
    while (Q--) {
        int x, u, v;
        scanf("%d", &x);
        if (x == 1) {
            scanf("%d %d", &u, &v);
            occupy(u, v);
        } else if (x == 2) {
            scanf("%d", &u);
            modify(1, 1, n, dfn[u], dfn[u], 0);
        } else {
            scanf("%d", &u);
            modify(1, 1, n, dfn[u], dfn[u] + siz[u] - 1, 1);
        }
        printf("%d\n", sum[1]);
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}