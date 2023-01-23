#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 10005;

int n;
double a[N], b[N], c[N], e[N], k[N];
vector<int> G[N];

void dfs(int u, int fa) {
    double rea = 0, reb = 0, rec = 0;
    double m = 0;
    for (int v : G[u]) {
        m += 1.0;
        if (v == fa) continue;
        dfs(v, u);
        rea += a[v], reb += b[v], rec += c[v];
    }
    if (m == 1.0 && u != 1) {
        a[u]= k[u];
        b[u] = c[u] = 1 - k[u] - e[u];
        return;
    }
    a[u] = (k[u] + (1 - k[u] - e[u]) / m * rea) / (1 - (1 - k[u] - e[u]) / m * reb);
    b[u] =  (1 - k[u] - e[u]) / m / (1 - (1 - k[u] - e[u]) / m * reb);
    c[u] = (1 - k[u] - e[u]) / m * (rec + m) / (1 - (1 - k[u] - e[u]) / m * reb);
}

void rmain() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) G[i].clear();
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    for (int i = 1; i <= n; i++) scanf("%lf %lf", &k[i], &e[i]), k[i] /= 100.0, e[i] /= 100.0;
    dfs(1, 1);
    if (fabs(a[1] - 1.0) < 1e-15) puts("impossible");
    else printf("%.6lf\n", c[1] / (1 - a[1]));
}

int main() {
    int T, Case = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case %d: ", ++Case);
        rmain();
    }
    return 0;
}