#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 2005;

int n, A, B;
double dp[N][4], p[N][4];
bool vis[N][4];

double dfs(int x, int cur) {
    if (x > n) return 0;
    if (vis[x][cur]) return dp[x][cur];
    vis[x][cur] = 1;
    double &re = dp[x][cur];
    re = 0;
    if (cur == 1) {
        double sum = 1.0;
        for (int i = x + A; i <= min(x + B, n); i++) {
            re += sum * p[i][2] * (dfs(i, 2) + 1);
            re += sum * p[i][3] * (dfs(i, 3) + 1);
            sum *= (p[i][1] + p[i][0]);
        }
        if (x + B > n) re += sum * 1;
    } else if (cur == 2) {
        double sum = 1.0;
        for (int i = x + A; i <= min(x + B, n); i++) {
            re += sum * p[i][1] * (dfs(i, 1) + 1);
            re += sum * p[i][3] * (dfs(i, 3) + 1);
            sum *= (p[i][2] + p[i][0]);
        }
        if (x + B > n) re += sum * 1;
    } else {
        double sum = 1.0;
        for (int i = x + A; i <= min(x + B, n); i++) {
            re += sum * p[i][1] * (dfs(i, 1) + 1);
            re += sum * p[i][2] * (dfs(i, 2) + 1);
            re += sum * p[i][3] * (dfs(i, 3) + 1);
            sum *= p[i][0];
        }
        if (x + B > n) re += sum * 1;
    }
    return re;
}

void rmain() {
    memset(vis, 0, sizeof vis);
    scanf("%d %d %d", &n, &A, &B);
    for (int i = 1; i <= n; i++) 
        for (int j = 0; j < 4; j++) scanf("%lf", &p[i][j]);
    printf("%.8lf\n", dfs(0, 3));
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}