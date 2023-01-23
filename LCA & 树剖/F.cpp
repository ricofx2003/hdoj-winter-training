#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;

int n, m, a[N];
vector<int> G[N];
int top[N], dep[N], faz[N], siz[N], son[N];
int dfc, rnk[N], dfn[N];
#define lson (x<<1)
#define rson ((x<<1)|1)
#define mid (l + r >> 1)
int sum[N << 2];
void pushup(int x) { sum[x] = sum[lson] ^ sum[rson]; }
void build(int x, int l, int r) {
	if (l == r) {
    	sum[x] = a[rnk[l]];
    	return;
	}
	build(lson, l, mid);
	build(rson, mid + 1, r);
	pushup(x);
	return;
}
void modify(int x, int l, int r, const int pos, const int d) {
    if (l == r) {
        sum[x] = d;
        return;
    }
	if (pos <= mid) modify(lson, l, mid, pos, d);
    else modify(rson, mid + 1, r,pos, d);
	pushup(x);
	return;
}
ll query(int x, int l, int r, const int L, const int R) {
    if (r < L || l > R) return 0;
    if (l >= L && r <= R) return sum[x];
    return query(rson, mid + 1, r, L, R) ^ query(lson, l, mid, L, R);
}
#undef mid
#undef lson
#undef rson
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
void rmain() {
    dfc = 0;
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) G[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]), a[i]++;
    dfs1(1, 1);
    dfs2(1, 1, 1);
    build(1, 1, n);
    while (m--) {
        int op, u, v;
        scanf("%d %d %d", &op, &u, &v);
        if (op == 0) {
            v++;
            modify(1, 1, n, dfn[u], v);
        } else {
            int ans = 0;
            while (top[u] != top[v]) {
                if (dep[top[u]] > dep[top[v]]) {
                    ans ^= query(1, 1, n, dfn[top[u]], dfn[u]);
                    u = faz[top[u]];
                } else {
                    ans ^= query(1, 1, n, dfn[top[v]], dfn[v]);
                    v = faz[top[v]];
                }
            }
            if (dep[u] > dep[v]) {
                ans ^= query(1, 1, n, dfn[v], dfn[u]);
            } else {
                ans ^=query(1, 1, n, dfn[u], dfn[v]);
            }
            printf("%d\n", ans - 1);
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