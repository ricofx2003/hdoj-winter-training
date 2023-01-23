#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;


int n, m;
int dis[N], val[N], incnt[N];
bool instk[N];
bitset<2005> f[2005];
vector<int> G[N];
bool reachable(int start) {
    queue<int> q;
    memset(incnt, 0, sizeof incnt);
    memset(dis, -0x3f, sizeof dis);
    memset(instk, 0, sizeof instk);
    dis[start] = 100;
    instk[start] = 1;
    q.push(start);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        instk[u] = 0;
        incnt[u]++;
        for (int v : G[u]) {
            int w = val[v];
            if (dis[u] + w <= 0) continue;
            if (dis[v] < dis[u] + w) {
                dis[v] = dis[u] + w;
                if (!instk[v]) {
                    q.push(v), instk[v] = 1;
                    incnt[v]++;
                    if (incnt[v] > n && f[v][n]) return 1;
                    if (incnt[v] > n + 2) return 0;
                }
            }
        }
    }
    return dis[n] > 0;
}

void rmain() {
    for (int i = 1; i <= n; i++) {
        int num;
        G[i].clear();
        f[i].reset();
        f[i].set(i);
        scanf("%d %d", &val[i], &num);
        for (int j = 1; j <= num; j++) {
            int v;
            scanf("%d", &v);
            G[i].push_back(v);
            f[i].set(v);
        }
    }
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            if (f[i][k]) f[i] |= f[k];
        }
    }
    reachable(1) ? puts("winnable") : puts("hopeless");
}

int main() {
    while (~scanf("%d", &n) && n != -1) {
        rmain();
    }
    return 0;
}