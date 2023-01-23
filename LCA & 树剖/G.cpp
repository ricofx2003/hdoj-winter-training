#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 100005;

int n, val[N];
vector<int> G[N];
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
struct node {
    int len, lc, fc, rc, LC, FC, RC, lv, rv;
    void init(int x = 0) {
        lc = fc = rc = LC = RC = FC = len = 1;
        lv = rv = x;
    }
    void reverse() {
        swap(rc, LC);
        swap(lc, RC);
        swap(FC, fc);
        swap(lv, rv);
    }
    node operator + (const node &x) const {
        if (x.lc == 0) return *this;
        if (lc == 0) return x;
        node b;
        b.len = len + x.len;
        b.lv = lv;
        b.rv = x.rv;
        b.lc = (lc == len) && (x.lv > rv) ? lc + x.lc : lc;
        b.rc = (x.rc == x.len) && (x.lv > rv) ? rc + x.rc : x.rc;
        b.fc = max({fc, x.fc, (x.lv > rv ? x.lc + rc : 0)});
        b.LC = (LC == len) && (x.lv < rv) ? LC + x.LC : LC;
        b.RC = (x.RC == x.len) && (x.lv < rv) ? RC + x.RC : x.RC;
        b.FC = max({FC, x.FC, (x.lv < rv ? x.LC + RC : 0)});
        return b;
    }
} tr[N << 2];
#define lson (p << 1)
#define rson ((p << 1) | 1)
#define mid ((l + r) >> 1)
void build(int p, int l, int r) {
    if (l == r) {
        tr[p].init(val[rnk[l]]);
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    tr[p] = tr[lson] + tr[rson];
    return;
}
node query(int p, int l, int r, int L, int R) {
    if (l > R || r < L) return (node){0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (l >= L && r <= R) return tr[p];
    return query(lson, l, mid, L, R) + query(rson, mid + 1, r, L, R);
}
int query(int u, int v) {
    node Du = (node){0, 0, 0, 0, 0, 0, 0, 0, 0}, Dv = (node){0, 0, 0, 0, 0, 0, 0, 0, 0};
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]]) {
            Du = query(1, 1, n, dfn[top[u]], dfn[u]) + Du;
            u = faz[top[u]];
        } else {
            Dv = query(1, 1, n, dfn[top[v]], dfn[v]) + Dv;
            v = faz[top[v]];
        }
    }
    if (dep[u] < dep[v]) {
        Du.reverse();
        Du = Du + query(1, 1, n, dfn[u], dfn[v]);
        Du = Du + Dv;
    } else {
        Du = query(1, 1, n, dfn[v], dfn[u]) + Du;
        Du.reverse();
        Du = Du + Dv;
    }
    return Du.fc;
}

void rmain() {
    dfc = 0;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &val[i]), G[i].clear();
    for (int i = 2, f; i <= n; i++) {
        scanf("%d", &f);
        G[f].push_back(i);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    build(1, 1, n);
    int Q;
    scanf("%d", &Q);
    while (Q--) {
        int u, v;
        scanf("%d %d", &u, &v);
        printf("%d\n", query(u, v));
    }
}

int main() {
    int T;
    scanf("%d", &T);
    for (int i = 1; i <= T; i++) {
        printf("Case #%d:\n", i);
        rmain();
        if (i != T) puts("");
    }
    return 0;
}