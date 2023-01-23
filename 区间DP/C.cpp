#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 10007;
const int N = 1005;
char s[N];
int n, dp[N][N];

void rmain() {
    scanf("%s", s + 1);
    n = strlen(s + 1);
    for (int len = 1; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = 0;
            if (s[i] == s[j]) {
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + 1) % mod;
            } else {
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + mod - dp[i + 1][j - 1]) % mod;
            }
        }
    }
    printf("%d\n", dp[1][n]);
}

int main() {
    int T, C = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case %d: ", ++C);
        rmain();
    }
    return 0;
}