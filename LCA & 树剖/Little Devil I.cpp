#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;

int n;
int head[N], ecnt;
struct Edge {
    int nxt, v;
} e[N * 2];
void add_edge(int u, int v) {
    e[++ecnt] = (Edge){head[u], v};
    head[u] = ecnt;
}
int top[N], dep[N], faz[N], siz[N], son[N];
int dfc, rnk[N], dfn[N];
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
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
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v;
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
#define lson (x<<1)
#define rson ((x<<1)|1)
#define mid (l + r >> 1)
struct node {
    int len, lv, rv, LV, RV, fc;
    void init() {
        len = 1;
        lv = rv = LV = RV = fc = 0;
    }
    void reverse() {
        swap(lv, rv);
        swap(LV, RV);
    }
    node operator + (const node &x) const {
        if (x.len == 0) return *this;
        if (len == 0) return x;
        node b;
        b.len = len + x.len;
        b.lv = lv;
        b.rv = x.rv;
        b.LV = LV;
        b.RV = x.RV;
        b.fc = fc + x.fc + ((RV + x.lv) & 1);
        return b;
    }
} tr[N << 2];
int tag[N << 2], TAG[N << 2];
void build(int x, int l, int r) {
    tag[x] = TAG[x] = 0;
    if (l == r) {
        tr[x].init();
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    tr[x] = tr[lson] + tr[rson];
}
void add(int x, int tp) {
    if (tp == 0) {
        tr[x].lv ^= 1;
        tr[x].rv ^= 1;
        tr[x].fc = tr[x].len - 1 - tr[x].fc;
        tag[x] ^= 1;
    } else {
        tr[x].RV ^= 1;
        tr[x].LV ^= 1;
        tr[x].fc = tr[x].len - 1 - tr[x].fc;
        TAG[x] ^= 1;
    }
}
void pushdown(int x) {
    if (tag[x]) {
        add(lson, 0);
        add(rson, 0);
        tag[x] = 0;
    }
    if (TAG[x]) {
        add(lson, 1);
        add(rson, 1);
        TAG[x] = 0;
    }
}
void modify(int x, int l, int r, int L, int R, int tp) {
    if (l > R || r < L) return;
    if (l >= L && r <= R) {
        add(x, tp);
        return;
    }
    pushdown(x);
    modify(lson, l, mid, L, R, tp);
    modify(rson, mid + 1, r, L, R, tp);
    tr[x] = tr[lson] + tr[rson];
}
node query(int x, int l, int r, int L, int R) {
    if (l > R || r < L) return (node){0, 0, 0, 0, 0, 0};
    if (l >= L && r <= R) return tr[x];
    pushdown(x);
    return query(lson, l, mid, L, R) + query(rson, mid + 1, r, L, R);
}
int query(int u, int v) {
    node Du, Dv;
    Du = Dv = (node){0, 0, 0, 0, 0, 0};
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
        Dv = query(1, 1, n, dfn[u], dfn[v]) + Dv;
        Dv.reverse();
        Du = Dv + Du;
    } else {
        Du = query(1, 1, n, dfn[v], dfn[u]) + Du;
        Du.reverse();
        Du = Du + Dv;
    }
    return Du.fc;
}

void rmain() {
    dfc = 0;
    ecnt = 0;
    memset(head, 0, sizeof head);
    scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u, v);
        add_edge(v, u);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    build(1, 1, n);
    int Q;
    scanf("%d", &Q);
    while (Q--) {
        int op, u, v;
        scanf("%d %d %d", &op, &u, &v);
        if (op == 1) {
            while (top[u] != top[v]) {
                if (dep[top[u]] > dep[top[v]]) {
                    modify(1, 1, n, dfn[top[u]], dfn[u], 0);
                    u = faz[top[u]];
                } else {
                    modify(1, 1, n, dfn[top[v]], dfn[v], 0);
                    v = faz[top[v]];
                }
            }
            if (dep[u] < dep[v]) {
                modify(1, 1, n, dfn[u], dfn[v], 0);
                modify(1, 1, n, dfn[u], dfn[u], 0);
            } else {    
                modify(1, 1, n, dfn[v], dfn[u], 0);
                modify(1, 1, n, dfn[v], dfn[v], 0);
            }
        } else if (op == 2) {
            while (top[u] != top[v]) {
                if (dep[top[u]] > dep[top[v]]) {
                    modify(1, 1, n, dfn[top[u]], dfn[u], 0);
                    modify(1, 1, n, dfn[top[u]], dfn[u], 1);
                    u = faz[top[u]];
                } else {
                    modify(1, 1, n, dfn[top[v]], dfn[v], 1);
                    modify(1, 1, n, dfn[top[v]], dfn[v], 0);
                    v = faz[top[v]];
                }
            }
            if (dep[u] < dep[v]) {
                modify(1, 1, n, dfn[u], dfn[v], 1);
                modify(1, 1, n, dfn[u], dfn[v], 0);
            } else {    
                modify(1, 1, n, dfn[v], dfn[u], 1);
                modify(1, 1, n, dfn[v], dfn[u], 0);
            }
        } else {
            printf("%d\n", query(u, v));
        }
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