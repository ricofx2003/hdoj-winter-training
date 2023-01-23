#include <bits/stdc++.h>
using namespace std;
using i64 = long long;

const double eps = 1e-9;
const int N = 1000 + 5;
const int inf = 0x3f3f3f3f;
int head[N], cur[N], ecnt, s, t, n, m;
double dis[N];
double p[11][1005];
double mincost;
bool vis[N];
struct Edge {
    int nxt, v, flow, cap;
    double w;
}e[500];
void add_edge(int u, int v, int flow, int cap, double w) {
    e[ecnt] = {head[u], v, flow, cap, w}; head[u] = ecnt++;
    e[ecnt] = {head[v], u, flow, 0, -w}; head[v] = ecnt++;
}
bool spfa(int s, int t) {
    memset(vis, 0, sizeof vis);
    fill(dis, dis + t + 1, 1e18);
    std::queue<int> q;
    q.push(s);
    dis[s] = 0;
    vis[s] = 1;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].v;
            if (e[i].flow < e[i].cap && dis[u] + e[i].w + eps < dis[v]) {
                dis[v] = dis[u] + e[i].w;
                if (!vis[v]) vis[v] = 1, q.push(v);
            }
        }
    }
    return fabs(dis[t] - 1e18) > eps;
}
int dfs(int u, int a) {
    if (vis[u]) return 0;
    if (u == t || !a) return a;
    vis[u] = 1;
    int b = 0;
    int flow = 0, f;
    for (int& i = cur[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (fabs(dis[u] + e[i].w - dis[v]) < eps && (f = dfs(v, std::min(a, e[i].cap - e[i].flow))) > 0) {
            e[i].flow += f;
            e[i ^ 1].flow -= f;
            flow += f;
            mincost += e[i].w * f;
            a -= f;
            if (!a) break;
        }
    }
    vis[u] = 0;
    return flow;
}

double solve(int l, int r) {
    memset(head, -1, sizeof head);
    ecnt = 0;
    mincost = 0;
    s = 0, t = n + r - l + 2;
    for (int i = 1; i <= n; i++) add_edge(s, i, 0, 1, 0);
    for (int i = n + 1; i < t; i++) add_edge(i, t, 0, 1, 0);
    for (int i = 1; i <= n; i++) {
        for (int j = l; j <= r; j++) add_edge(i, n + j - l + 1, 0, 1, -p[i][j]);
    }
    int ans = 0;
    while (spfa(s, t)) {
        for (int i = 0; i <= t; i++) cur[i] = head[i];
        ans += dfs(s, inf);
    }
    return mincost;
}

void rmain() {
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) scanf("%lf", &p[i][j]);
    }
    double ans = 0;
    for (int i = 1; i <= m; i += n) {
        ans -= solve(i, min(i + n - 1, m));
    }
    printf("%.5lf\n", ans);
}

int main() {
    int T, Case = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case #%d: ", ++Case);
        rmain();
    }
    return 0;
}