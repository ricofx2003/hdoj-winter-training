#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;
#define int long long
int n;
int head[N], ecnt;
int ans, loo;
struct node {
    int d, w;
} d1[N], d2[N];
struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}
void dfs(int u, int fa) {
    d1[u] = (node){1, 0};
    d2[u] = (node){0, 0};
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        dfs(v, u);
        int t = d1[v].w + w;
        if (t > d1[u].w) {
            d2[u].w = d1[u].w;
            d2[u].d = d1[u].d;
            d1[u].w = t;
            d1[u].d = d1[v].d;
        } else if (t == d1[u].w) {
            d1[u].d += d1[v].d;
        } else if (t > d2[u].w) {
            d2[u].w = t;
            d2[u].d = d1[v].d;
        } else if (t == d2[u].w) {
            d2[u].d += d1[v].d;
        }
    }
    int re = 0;
    int sum = 0;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        if (d1[v].w + w == d1[u].w) sum += d1[v].d;
    }
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        if (d1[v].w + w == d1[u].w) {
            sum -= d1[v].d;
            re += d1[v].d * sum;
        }
    }
    if (re != 0 && d1[u].w * 2 >= loo) {
        if (d1[u].w * 2 > loo) loo = d1[u].w + d1[u].w,ans = re;
        else ans += re;
    }
    if (re == 0 && d2[u].d != 0 && d2[u].w + d1[u].w >= loo) {
        if (d2[u].w + d1[u].w > loo) loo = d2[u].w + d1[u].w, ans = d2[u].d * d1[u].d;
        else ans += d2[u].d * d1[u].d;
    }
}
void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    ans = 0;
    loo = 0;
    for (int i = 1; i < n; i++) {
        int u, v, w;
        scanf("%lld %lld %lld", &u, &v, &w);
        add_edge(u, v, w);
    }
    dfs(1, 1);
    printf("%lld %lld\n", loo, ans);
}
signed main() {
    while (~scanf("%lld", &n)) {
        rmain();
    }
    return 0;
}