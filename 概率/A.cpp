#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1005;

int n, m;
double dp[N][N];
double p[N][N][3];

void rmain() {  
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            for (int x = 0; x < 3; x++) scanf("%lf", &p[i][j][x]);
        }
    memset(dp, 0, sizeof dp);
    for (int i = n; i; i--) {
        for (int j = m; j; j--) {
            if ((i == n && j == m) || p[i][j][0] == 1.0) continue;
            dp[i][j] = (dp[i + 1][j] * p[i][j][2] + dp[i][j + 1] * p[i][j][1] + 2) / (1 - p[i][j][0]);
        }
    }
    printf("%.3lf\n", dp[1][1]);
}
int main() {
    while (~scanf("%d %d", &n, &m)) {
        rmain();
    }
    return 0;
}