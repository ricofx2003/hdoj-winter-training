#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1000005;


struct node {
    int v, w;
};
int n, m, dis[N];
vector<node> G[N];
bool vis[N];
void Dijstra(int start) {
    memset(dis, 0x3f, sizeof dis);
    memset(vis, 0, sizeof vis);
    dis[start] = 0;
    priority_queue<pair<int, int>> q;
    q.push({0, start});
    while (!q.empty()) {
        int u = q.top().second;
        q.pop();
        if (vis[u]) continue;
        for (auto x : G[u]) {
            int v = x.v, w = x.w;
            if (dis[v] > dis[u] + w) {
                dis[v] = dis[u] + w;
                q.push({-dis[v], v});
            }
        }
    }
}

void rmain() {
    scanf("%d %d", &n, &m);
    vector<array<int, 3>> Edge(m + 1);
    for (int i = 1; i <= m; i++) {
        scanf("%d %d %d", &Edge[i][0], &Edge[i][1], &Edge[i][2]);
    }
    for (int i = 1; i <= n; i++) G[i].clear();
    for (auto x : Edge) {
        int u = x[0], v = x[1], w = x[2];
        G[u].push_back({v, w});
    }
    Dijstra(1);
    ll ans = 0;
    for (int i = 2; i <= n; i++) ans += dis[i];
    for (int i = 1; i <= n; i++) G[i].clear();
    for (auto x : Edge) {
        int u = x[1], v = x[0], w = x[2];
        G[u].push_back({v, w});
    }
    Dijstra(1);
    for (int i = 2; i <= n; i++) ans += dis[i];
    printf("%lld\n", ans);
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}