#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 55;
#define int long long
int dp[N][100005];
int f[100005], siz[N];
int n, m, p[N], c, v;

void rmain() {
    for (int k = 1; k <= n; k++) {
        scanf("%lld", &p[k]);
        memset(dp[k], 0, sizeof dp[k]);
        siz[k] = 0;
        int num;
        scanf("%lld", &num);
        for (int j = 1; j <= num; j++) {
            scanf("%lld %lld", &c, &v);
            siz[k] += c;
            for (int i = siz[k]; i >= c; i--) {
                dp[k][i] = max(dp[k][i], dp[k][i - c] + v);
            }
        }
    }
    memset(f, 0, sizeof f);
    int S = 0;
    for (int k = 1; k <= n; k++) {
        S += siz[k] + p[k];
        if (S > m) S = m;
        for (int i = S; i >= p[k]; i--) {
            for (int j = p[k]; j <= p[k] + siz[k] && j <= i; j++) {
                f[i] = max(f[i], dp[k][j - p[k]] + f[i - j]);
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= m; i++) ans = max(ans, f[i]);
    printf("%lld\n", ans);
}
signed main() {
    while (~scanf("%lld %lld", &n, &m)) {
        rmain();
    }
    return 0;
}