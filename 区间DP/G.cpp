/*#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 5201314, M = 11731;
const int N = 505;
int n;
char s[N][N];
int H(int a, int b, int c, int d) {
    //return (a - 1) * n + b + ((c - 1) * n + d) * n * n;
    return ((a - 1) * n % M + b + ((c - 1) * n % M + d) * n % M * n % M) % M;
}
int head[M + 1], ecnt;
struct E {
    int nxt, a, b, c, d, w;
} e[83083501];
void add(int a, int b, int c, int d, int u) {
    e[++ecnt] = {head[u], a, b, c, d, 0};
    head[u] = ecnt;
}
int find(int a, int b, int c, int d, int u) {
    for (int i = head[u]; i; i = e[i].nxt) {
        if (e[i].a == a && e[i].b == b && e[i].c == c && e[i].d == d) return i;
    }
    return 0;
}

void add(int &a, int b) {
    ((a += b) >= mod) && (a -= mod);
}
int dfs(int a, int b, int c, int d) {
    if (a > c || b > d) return 0;
    if (a == c && b == d) return 1;
    int now = H(a, b, c, d), cu = find(a, b, c, d, now);
    if (cu) return e[cu].w;
    add(a, b, c, d, now);
    int &re = e[ecnt].w;
    re = 0;
    if (a == c) {
        if (b + 1 == d) return re = s[a][b] == s[c][d];
        else return re = dfs(a, b + 1, c, d - 1);
    }
    if (b == d) {
        if (a + 1 == c) return re = s[a][b] == s[c][d];
        else return re = dfs(a + 1, b, c - 1, d);
    }
    if (s[a + 1][b] == s[c - 1][d]) add(re, dfs(a + 1, b, c - 1, d));
    if (s[a + 1][b] == s[c][d - 1]) add(re, dfs(a + 1, b, c, d - 1));
    if (s[a][b + 1] == s[c - 1][d]) add(re, dfs(a, b + 1, c - 1, d));
    if (s[a][b + 1] == s[c][d - 1]) add(re, dfs(a, b + 1, c, d - 1));
    return re;
}
void rmain() {
    memset(head, 0, sizeof head);
    ecnt = 0;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%s", s[i] + 1);
    if (s[1][1] != s[n][n]) printf("%d\n", 0);
    else printf("%d\n", dfs(1, 1, n, n));
}

int main() {
    freopen("input.txt", "r", stdin);
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}*/
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 5201314;
const int N = 505;

int n;
char s[N][N];
int dp[2][N][N];

void add(int &a, int b) {
    ((a += b) >= mod) && (a -= mod);
}

void rmain() {
    memset(dp, 0, sizeof dp);
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        scanf("%s", s[i] + 1);
    }
    int nw = 1, od = 0;
    dp[nw][1][n] = (s[1][1] == s[n][n]);
    for (int len = 1; len <= n; len++) {
        nw ^= 1, od ^= 1;
        for (int a = 1; a <= len; a++) {
            for (int c = n; c >= n - len + 1; c--) {
                dp[nw][a][c] = 0;
                int b = len - a + 1, d = 2 * n - len - c + 1;
                if (s[a][b] != s[c][d]) continue;
                add(dp[nw][a][c], dp[od][a][c]);
                add(dp[nw][a][c], dp[od][a - 1][c + 1]);
                add(dp[nw][a][c], dp[od][a][c + 1]);
                add(dp[nw][a][c], dp[od][a - 1][c]);
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; i++) add(ans, dp[nw][i][i]);
    printf("%d\n", ans);
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}