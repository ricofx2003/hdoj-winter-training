#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 105;
int dp[N][N], n, a[N], sum[N];
void rmain() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
        scanf("%d", &a[i]), sum[i] = sum[i - 1] + a[i];
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = 0x3f3f3f3f;
            for (int k = i; k <= j; k++) {
                dp[i][j] = min(dp[i][j], dp[i + 1][k] + (k - i) * a[i] + (sum[j] - sum[k]) * (k - i + 1) + dp[k + 1][j]);
            }
        }
    }
    printf("%d\n", dp[1][n]);
}
int main() {
    int T, C = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case #%d: ", ++C);
        rmain();
    }
    return 0;
}