#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e6 + 5;

int T, n, op;
double m, f[N], g[N], sf[N], sg[N];

void rmain() {
    while (T--) {
        scanf("%d %lf %d", &op, &m, &n);
        if (op == 0) {
            f[0] = sf[0] = 1.0;
            for (int i = 1; i < n; i++) {
                f[i] = 1.0 + (m - 1.0) * (sf[i - 1]);
                sf[i] = sf[i - 1] + f[i];
            }
            printf("%.9lf\n", sf[n - 1]);
        } else {
            g[0] = sg[0] = sf[0] = 1.0;
            for (int i = 1; i < n; i++) {
                g[i] = (m + i * 1.0 * sg[i - 1] - sf[i - 1]) / (m - i * 1.0);
                sg[i] = sg[i - 1] + g[i];
                sf[i] = sf[i - 1] + sg[i];
            }
            printf("%.9lf\n", sg[n - 1]);
        }
    }
}
int main() {
    while (~scanf("%d", &T)) {
        rmain();
    }
    return 0;
}