#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 35;

int v[31][31][1801];
int n, m, a[N][N];

void rmain() {
    scanf("%d %d", &n, &m);
    memset(v, 0x3f, sizeof v);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
             scanf("%d", &a[i][j]);
        }
    v[1][1][a[1][1]] = a[1][1] * a[1][1];
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            for (int k = a[i][j]; k <= (i + j - 1) * 30; k++) {
                if (j > 1 && v[i][j - 1][k - a[i][j]] != 0x3f3f3f3f) v[i][j][k] = min(v[i][j - 1][k - a[i][j]] + a[i][j] * a[i][j], v[i][j][k]);
                if (i > 1 && v[i - 1][j][k - a[i][j]] != 0x3f3f3f3f) v[i][j][k] = min(v[i - 1][j][k - a[i][j]] + a[i][j] * a[i][j], v[i][j][k]);
            }
        }
    }
    ll ans = 1e18;
    for (int k = a[n][m]; k <= (n + m - 1) * 30; k++) {
        if (v[n][m][k] != 0x3f3f3f3f) ans = min(ans, v[n][m][k] * 1ll * (n + m - 1) - k * k);
    }
    printf("%lld\n", ans);  
}

int main() {
    int T, Case = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case #%d: ", ++Case);
        rmain();
    }
    return 0;
}