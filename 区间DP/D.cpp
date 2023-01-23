#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1005;
int n, a[N * 2];
int dp[N * 2][N * 2];
void rmain() {
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]), a[i + n] = a[i];
    for (int len = 1; len <= n; len++) {
        for (int i = 1; i + len - 1 <= 2 * n; i++) {
            int j = i + len - 1;
            if (a[i] == a[j]) dp[i][j] = max({dp[i + 1][j], dp[i][j - 1], dp[i + 1][j - 1] + (i == j ? 1 : 2)}); 
            else dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) {
        ans = max(ans, dp[i][i + n - 1]);
        ans = max(ans, dp[i + 1][i + n - 1] + 1);
    }
    printf("%d\n", ans);
}

int main() {
    while (~scanf("%d", &n) && n) {
        rmain();
    }
    return 0;
}