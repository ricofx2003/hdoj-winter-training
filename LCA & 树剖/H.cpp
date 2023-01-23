#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}

int maxx[N << 2];
int n, val[N];
int top[N], dep[N], faz[N], siz[N], son[N];
int dfc, rnk[N], dfn[N];
#define lson (x<<1)
#define rson ((x<<1)|1)
#define mid (l + r >> 1)
void build(int x, int l, int r) {
    if (l == r) {
        maxx[x] = val[rnk[l]];
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    maxx[x] = max(maxx[lson], maxx[rson]);
}
void modify(int x, int l, int r, int pos, int d) {
    if (l == r) {
        maxx[x] = d;
        return;
    }
    if (pos <= mid) modify(lson, l, mid, pos, d);
    else modify(rson, mid + 1, r, pos, d);
    maxx[x] = max(maxx[lson], maxx[rson]);
}
int query(int x, int l, int r, int L, int R) {
    if (r < L || l > R) return -1;
    if (l >= L && r <= R) return maxx[x];
    return max(query(lson, l, mid, L, R), query(rson, mid + 1, r, L, R));
}
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        val[v] = w;
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
        int v = e[i].v, w = e[i].w;
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
struct node {
    int x, y;
} inp[N];
int p[N], Ans[N], pv[N];
void rmain() {
    dfc = 0;
    val[1] = -1;
    memset(head, 0, sizeof head);
    ecnt = 0;
    scanf("%d", &n);
    for (int i = 1; i < n; i++) {
        int u, v , w;
        scanf("%d %d %d", &u, &v, &w);
        add_edge(u, v, w);
    }
    dfs1(1, 1);
    dfs2(1, 1, 1);
    build(1, 1, n);
    int Q;
    scanf("%d", &Q);
    iota(p + 1, p + Q + 1, 1);
    iota(pv + 1, pv + n + 1, 1);
    for (int i = 1; i <= Q; i++) scanf("%d %d", &inp[i].x, &inp[i].y);
    sort(p + 1, p + Q + 1, [](int x, int y) {
        return inp[x].y > inp[y].y;
    });
    sort(pv + 1, pv + n + 1, [](int x, int y) {
        return val[x] > val[y];
    });
    int j = 1;
    for (int i = 1; i <= Q; i++) {
        int nv = inp[p[i]].y;
        while (j < n && val[pv[j]] > nv) {
            modify(1, 1, n, dfn[pv[j]], -1);
            j++;
        }
        int u = inp[p[i]].x, v = 1, ans = -1;
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                ans = max(ans, query(1, 1, n, dfn[top[u]], dfn[u]));
                u = faz[top[u]];
            } else {
                ans = max(ans, query(1, 1, n, dfn[top[v]], dfn[v]));
                v = faz[top[v]];
            }
        }
        ans = max(ans, query(1, 1, n, dfn[v], dfn[u]));
        Ans[p[i]] = ans;
    }
    for (int i = 1; i <= Q; i++) printf("%d\n", Ans[i]);
}
int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}