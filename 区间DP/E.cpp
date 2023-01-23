#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 305;

int gcd(int a, int b) {
    return (b == 0) ? a : gcd(b, a % b);
}
int n, a[N], b[N], g[N][N];
ll sum[N], dp[N][N];
void rmain() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", a + i);
    for (int i = 1; i <= n; i++) scanf("%d", b + i), sum[i] = sum[i - 1] + b[i];
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) g[i][j] = gcd(a[i], a[j]) == 1 ? 0 : b[i] + b[j];
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = 0;
            if (g[i][j] && dp[i + 1][j - 1] == sum[j - 1] - sum[i]) dp[i][j] = g[i][j] + dp[i + 1][j - 1];
            for (int k = i; k < j; k++) {
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j]);
            }
        }
    }
    printf("%lld\n", dp[1][n]);
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}