#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 100005;

int n, m;
vector<int> to[N];
double dp[N];
void add(int pos, int x, double p) {
    if(pos + x > n) dp[pos] += 1.0 / p;
    else dp[pos] += (dp[pos + x] + 1) / p;
}
void rmain() {
    for (int i = 1; i <= n; i++) to[i].clear();
    for (int i = 1; i <= m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        u++, v++;
        to[u].push_back(v);
    }
    dp[n + 1] = 0;
    for (int i = n; i >= 1; i--) {
        dp[i] = 0;
        int siz = to[i].size();
        if (siz == 0) for (int x = 1; x <= 6; x++) add(i, x, 6.0);
        else {
            for (int v : to[i]) {
                dp[i] += dp[v] / (siz);
            }
        }
    }
    printf("%.4lf\n", dp[1]);
}
int main() {
    while (~scanf("%d %d", &n, &m) && (n + m)) {
        rmain();
    }
    return 0;
}