#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 105;
char a[N], b[N];
int n, dp[N][N];
int trans[27][N][N];

int dfs(int c, int l, int r) {
    int &re = trans[c][l][r];
    if (re != -1) return re;
    if (l == r) return re = (c != b[l] - 'a');
    re = 0x3f3f3f3f;
    for (int i = l; i < r; i++) re = min(dfs(c, l, i) + dfs(c, i + 1, r), re);
    //for (int cc = 0; cc < 26; cc++) if (c != cc) re = min(re, dfs(cc, l, r) + 1);
    if (b[r] - 'a' != c) re = min(re, dfs(b[r] - 'a', l, r) + 1);
    if (b[l] - 'a' != c) re = min(re, dfs(b[l] - 'a', l, r) + 1);
    return re;
}

void rmain() {
    memset(trans, -1, sizeof trans);
    n = strlen(a + 1);
    for (int i = 1; i <= n; i++) dp[i][i] = (a[i] != b[i]);
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = dp[i + 1][j - 1] + dp[i][i] + dp[j][j];
            dp[i][j] = min(dp[i][j], dfs(b[i] - 'a', i, j) + 1);
            dp[i][j] = min(dp[i][j], dfs(b[j] - 'a', i, j) + 1);
            for (int k = i; k < j; k++) dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j]);
        }
    }
    printf("%d\n", dp[1][n]);
}
int main() {
    while (~scanf("%s %s", a + 1, b + 1)) {
        rmain();
    }
    return 0;
}