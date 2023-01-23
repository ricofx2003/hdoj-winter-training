#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 100 + 5;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 2];
void add(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
}

int n, m;
int dis[N], qcnt[N];
bool inq[N];
bool BellmanFord(int start) {
    queue<int> q;
    memset(dis, -0x3f, sizeof dis);
    memset(qcnt, 0, sizeof qcnt);
    memset(inq, 0, sizeof inq);
    q.push(start);
    dis[start] = 0;
    inq[start] = 1;
    qcnt[start]++;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = 0;
        for (int i = head[u]; i; i = e[i].nxt) {
            int v = e[i].v, w = e[i].w;
            if (dis[v] < dis[u] + w) {
                dis[v] = dis[u] + w;
                if (!inq[v]) {
                    q.push(v);
                    inq[v] = 1;
                    qcnt[v]++;
                    if (qcnt[v] > n) return 0;
                }
            }
        }
    }
    return 1;
}
void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    for (int i = 1; i <= m; i++) {
        int l, len, k;
        static char s[5];
        scanf("%d %d %s %d", &l, &len, s + 1, &k);
        if (s[1] == 'g') {
            k++;
            // s[i + len] - s[i - 1] >= k
            add(l - 1, l + len, k);
        } else {
            k--;
            // s[i + len] - s[i - 1] <= k
            add(l + len, l - 1, -k);
        }
    }
    // 1 2 3 >= 1
    //   2 3 4 <= 1
    bool flag = 1;
    for (int i = 0; i <= n; i++) {
        if (!BellmanFord(i)) {
            flag = 0;
        }
    }
    !flag ? puts("successful conspiracy") : puts("lamentable kingdom");
}

int main() {
    while (~scanf("%d %d", &n, &m) && n) {
        rmain();
    }
    return 0;
}