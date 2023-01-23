#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 105;

int n, m, dp[N], a[N][N];

void rmain() {
    memset(dp, 0, sizeof dp);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) scanf("%d", &a[i][j]);
    }
    for (int i = 1; i <= n; i++) {
        for (int j = m; j >= 0; j--) {
            for (int k = 1; k <= j; k++) {
                dp[j] = max(dp[j], dp[j - k] + a[i][k]);
            }
        }
    }
    printf("%d\n", dp[m]);
}
int main() {
    while (~scanf("%d %d", &n, &m) && (n + m)) {
        rmain();
    }
    return 0;
}