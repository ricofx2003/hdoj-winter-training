#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 500 + 5;


int n, m;
int in[N];
vector<int> G[N];
// int dis[N], qcnt[N];
// bool inq[N];
// bool BellmanFord(int start) {
//     queue<int> q;
//     memset(dis, -0x3f, sizeof dis);
//     memset(qcnt, 0, sizeof qcnt);
//     memset(inq, 0, sizeof inq);
//     q.push(start);
//     dis[start] = 0;
//     inq[start] = 1;
//     qcnt[start]++;
//     while (!q.empty()) {
//         int u = q.front();
//         q.pop();
//         inq[u] = 0;
//         for (int i = head[u]; i; i = e[i].nxt) {
//             int v = e[i].v, w = e[i].w;
//             if (dis[v] < dis[u] + w) {
//                 dis[v] = dis[u] + w;
//                 if (!inq[v]) {
//                     q.push(v);
//                     inq[v] = 1;
//                     qcnt[v]++;
//                     if (qcnt[v] > n) return 0;
//                 }
//             }
//         }
//     }
//     return 1;
// }
void rmain() {
    for (int i = 1; i <= n; i++) G[i].clear(), in[i] = 0;
    for (int i = 1; i <= m; i++) {
        int a, b;
        scanf("%d %d", &a, &b);
        G[a].push_back(b);
        in[b]++;
    }
    priority_queue<int> q;
    for (int i = 1; i <= n; i++) if (!in[i]) q.push(-i);
    vector<int> topo;
    while (!q.empty()) {
        int u = q.top();
        q.pop();
        topo.push_back(-u);
        for (int v : G[-u]) {
            in[v]--;
            if (!in[v]) q.push(-v);
        }
    }
    for (int i = 0; i < n; i++) {
        printf("%d%c", topo[i], " \n"[i == n -1]);
    }
}

int main() {
    while (~scanf("%d %d", &n, &m)) {
        rmain();
    }
    return 0;
}