#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 2500;

int n, m;
vector<int> G[N * 2], e[N * 2];
bool mark[N * 2];
pair<int, int> a[N];
int stk[N * 2], top;
bool dfs(int u) {
    if (mark[u ^ 1]) return false;
    if (mark[u]) return true;
    mark[u] = 1;
    stk[++top] = u;
    for (int v : G[u]) {
        if (!dfs(v)) return false;
    }
    return true;
}
bool two_sat(int mid) {
    fill(mark + 1, mark + n * 4 + 2, 0);
    for (int i = 1; i <= n * 4 + 1; i++) G[i] = e[i];
    for (int i = 1; i <= mid; i++) {
        int u = a[i].first, v = a[i].second;
        G[u * 2].push_back(v * 2 + 1);
        if (u != v) G[v * 2].push_back(u * 2 + 1);
    }
    for (int i = 1; i <= 2 * n; i++) {
        if (!mark[i * 2] && !mark[i * 2 + 1]) {
            top = 0;
            if (!dfs(2 * i + 1)) {
                while (top) mark[stk[top--]] = 0;
                if (!dfs(2 * i)) return 0;
            }
        }
    }
    return 1;
}

void rmain() {
    for (int i = 1; i <= n * 4 + 1; i++) e[i].clear();
    for (int i = 1; i <= n; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        u++, v++;
        e[u * 2 + 1].push_back(v * 2);
        e[v * 2 + 1].push_back(u * 2);
    }
    for (int i = 1; i <= m; i++) {
        scanf("%d %d", &a[i].first, &a[i].second);
        a[i].first++, a[i].second++;
    } 
    int l = 0, r = m, ans = -1;
    while (l <= r) {
        int mid = (l + r) >> 1;
        if (two_sat(mid)) l = mid + 1, ans = mid;
        else r = mid - 1;
    }
    printf("%d\n", ans);
}

int main() {
    while (~scanf("%d %d", &n, &m) && (n + m)) {
        rmain();
    }
    return 0;
}