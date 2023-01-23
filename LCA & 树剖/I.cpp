#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;

int n, m;
int head[N], ecnt;
struct Edge {
    int nxt, v;
} e[2000005];
void add_edge(int u, int v) {
    e[++ecnt] = (Edge){head[u], v};
    head[u] = ecnt;
}
int top[N], dep[N], faz[N], siz[N], son[N], wh[N];
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
int mx[N << 2];
void build(int x, int l, int r) {
    if (l == r) {
        mx[x] = 0;
        wh[l] = x;
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    mx[x] = 0;
}
void modify(int x, int d) {
    mx[x] += d;
    while (x > 1) {
        x >>= 1;
        mx[x] = max(mx[lson], mx[rson]);
    }
}
int query(int x, int l, int r) {
    while (l < r) {
        if (mx[lson] == mx[1]) r = mid, x = lson;
        else l = mid + 1, x = rson;
    }
    return l;
}
int Ans[N];
void rmain() {
    dfc = 0;
    ecnt = 0;
    memset(head, 0, sizeof head);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u, v);
        add_edge(v, u);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    int mc = 100000;
    ecnt = 0;
    memset(head, 0, sizeof head);
    for (int i = 1; i <= m; i++) {
        int u, v, x;
        scanf("%d %d %d", &u, &v, &x);
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                add_edge(dfn[top[u]], x);
                add_edge(dfn[u] + 1, -x);
                u = faz[top[u]];
            } else {
                add_edge(dfn[top[v]], x);
                add_edge(dfn[v] + 1, -x);
                v = faz[top[v]];
            }
        }
        if (dep[u] < dep[v]) {
            add_edge(dfn[u], x);
            add_edge(dfn[v] + 1, -x);
        } else {
            add_edge(dfn[v], x);
            add_edge(dfn[u] + 1, -x);
        }
    }
    build(1, 1, mc);
    for (int l = 1, r = 1; l <= n; l = r + 1) {
        while (r < n && dfn[top[rnk[r + 1]]] == l) r++;
        for (int i = l; i <= r; i++) {
            for (int lll = head[i]; lll; lll = e[lll].nxt) { 
                int x = e[lll].v;
                x > 0 ? modify(wh[x], 1) : modify(wh[-x], -1);
            }
            if (mx[1] == 0) Ans[rnk[i]] = 0;
            else Ans[rnk[i]] = query(1, 1, mc);
        }
    }
    for (int i = 1; i <= n; i++) printf("%d\n", Ans[i]);
}
int main() {
    while (~scanf("%d %d", &n, &m) && (n || m)) {
        rmain();
    }
    return 0;
}