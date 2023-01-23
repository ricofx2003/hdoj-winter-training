#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 10005;

int n, dw[N], up[N];
int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 1];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u, w}, head[v] = ecnt;
}
void dfs1(int u, int fa) {
    dw[u] = 0;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        dfs1(v, u);
        dw[u] = max(dw[v] + w, dw[u]);
    }
}
pair<int, int> stk[N];
int top;
void dfs2(int u, int fa) {
    int pre = 0, suf = 0;
    top = 0;
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v == fa) continue;
        stk[++top] = {v, w};
        up[v] = up[u] + w;
        up[v] = max(up[v], pre + w);
        pre = max(pre, dw[v] + w);
    }
    while (top) {
        int v = stk[top].first, w = stk[top].second;
        top--;
        up[v] = max(suf + w, up[v]);
        suf = max(suf, dw[v] + w);
    }
    for (int i = head[u]; i; i = e[i].nxt) {
        int v = e[i].v, w = e[i].w;
        if (v != fa) dfs2(v, u);
    }
}

void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    for (int i = 2; i <= n; i++) {
        int u, w;
        scanf("%d %d", &u, &w);
        add_edge(i, u, w);
    }
    dfs1(1, 1);
    dfs2(1, 1);
    for (int i = 1; i <= n; i++) printf("%d\n", max(dw[i], up[i]));
}
int main() {
    while (~scanf("%d", &n)) {
        rmain();
    }
    return 0;
}