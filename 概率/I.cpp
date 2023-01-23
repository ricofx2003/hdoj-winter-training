#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1005;

int n, m;
double dp[N][N], a[N][N];

void rmain() {
    for (int i = 1; i <= n; i++) {
        a[i][0] = 0;
        for (int j = 1; j <= m; j++) scanf("%lf", &a[i][j]), a[i][0] += a[i][j];
    }
    double ans = 0;
    for (int k = 1; k <= m; k++) {
        double sum = 0;
        for (int i = 1; i <= n; i++) sum += a[i][k] / a[i][0], ans += (a[i][k] / a[i][0]) * (1 -a[i][k] / a[i][0]);
        ans += sum * sum;
    }
    printf("%.2lf\n", ans);
}
int main() {
    while (~scanf("%d %d", &n, &m)) {
        rmain();
    }
    return 0;
}