#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 2005;

int n, m, k;
double f[N], p[5], c[N], g[N];

void rmain() {
    double p1 = 1 - p[1], p2 = p[2] / p1, p3 = p[3] / p1, p4 = p[4] / p1;
    f[1] = g[1] = p[4] / (1 - p[1] - p[2]);
    for (int i = 2; i <= n; i++) {
        double re = 0, re1 = 1;
        for (int j = 1; j <= i; j++) c[j] = (j <= k ? p4 : 0) + g[j - 1] * p3;
        for (int j = 1; j <= i; j++) re = p2 * re + c[j], re1 *= p2;
        f[i] = re / (1 - re1);
        f[1] = p4 + p2 * f[i];
        for (int j = 2; j < i; j++) {
            f[j] = p2 * f[j- 1] + p3 * g[j - 1] + (j <= k ? p4 : 0);
        }
        memcpy(g, f, sizeof g);
    }
    printf("%.5lf\n", f[m]);
}
int main() {
    while (~scanf("%d %d %d %lf %lf %lf %lf", &n, &m, &k, &p[1], &p[2], &p[3], &p[4])) {
        if (p[4] < 1e-6) puts("0.00000");
        else rmain();
    }
    return 0;
}