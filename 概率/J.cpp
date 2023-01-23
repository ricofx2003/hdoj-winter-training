#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 35;

int n;
double t;
double pow2[N], dp[N];

void rmain() {
    dp[n] = pow2[n];
    for (int i = n - 1; i >= 0; i--) {
        double m = max(t, pow2[i] / dp[i + 1]);
        double p = (m - t) / (1 - t);
        dp[i] = p * pow2[i] + (1 - p) * dp[i + 1] * (1 + m) / 2.0;
    }
    printf("%.3lf\n", dp[0]);
}
int main() {
    pow2[0] = 1.0;
    for (int i = 1; i < N; i++) pow2[i] = pow2[i - 1] * 2.0;
    while (~scanf("%d %lf", &n, &t) && n) {
        rmain();
    }
    return 0;
}