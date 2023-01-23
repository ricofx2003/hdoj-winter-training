#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 8000 * 2 + 5;

int n, m;
vector<int> G[N * 2];
bool mark[N * 2];
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
bool tow_sat() {
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
    fill(mark + 1, mark + 4 * n + 5, 0);
    for (int i = 1; i <= n * 4 + 5; i++) G[i].clear();
    for (int i = 1; i <= m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        G[u * 2 + 1].push_back(v * 2);
        G[v * 2 + 1].push_back(u * 2);
    }
    for (int i = 1; i <= 2 * n; i += 2) {
        int u = i, v = i + 1;
        G[u * 2 + 1].push_back(v * 2);
        G[v * 2 + 1].push_back(u * 2);
        G[u * 2].push_back(v * 2 + 1);
        G[v * 2].push_back(u * 2 + 1);
    }
    if (!tow_sat()) puts("NIE");
    else {
        for (int i = 1; i <= 2 * n; i += 2) {
            if (mark[i * 2 + 1]) printf("%d\n", i);
            else printf("%d\n", i + 1);
        }
    }
}

int main() {
    while (~scanf("%d %d", &n, &m)) {
        rmain();
    }
    return 0;
}