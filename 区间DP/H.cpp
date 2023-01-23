#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1e9 + 7;
const int N = 105;

int fpow(int a, int b) {
    int res = 1;
    for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1) res = res * 1ll * a % mod;
    return res;
}
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b) {
        ll d = exgcd(b, a % b, y, x);
        return y -= a / b * x, d;
    } else return x = 1, y = 0, a;
}
int getinv(int v) {
    return fpow(v, mod - 2);
    // ll x, y;
    // exgcd(v, mod, x, y);
    // return (x % mod + mod) % mod;
}
int fac[N], ifac[N];
void init_binom(int n) {
    fac[0] = ifac[0] = 1;
    for (int i = 1; i <= n; i++) fac[i] = fac[i - 1] * 1ll * i % mod;
    ifac[n] = getinv(fac[n]);
    for (int i = n; i > 1; i--) ifac[i - 1] = ifac[i] * 1ll * i % mod;
}
int binom(int a, int b) {
    if (b < 0 || a < 0 || b > a) return 0;
    return fac[a] * 1ll * ifac[b] % mod * ifac[a - b] % mod; 
}

void add(int &a, int b) {
    ((a += b) >= mod) && (a -= mod);
}
void dec(int &a, int b) {
    add(a, mod - b);
}
int n, a[N], dp[N][N];
char s[N];
void rmain() {
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]), dp[i][i] = a[i];
    scanf("%s", s + 1);
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i + len - 1 <= n; i++) {
            int j = i + len - 1;
            dp[i][j] = 0;
            for (int k = i; k < j; k++) {
                int left = fac[k - i] * 1ll * binom(j - i - 1, k - i) % mod;
                int right = fac[j - k - 1] * 1ll * binom(j - i - 1, k - i) % mod; 
                if (s[k] == '+') add(dp[i][j], (right * 1ll * dp[i][k] % mod + left * 1ll * dp[k + 1][j] % mod) % mod);
                else if (s[k] == '-') add(dp[i][j], (right * 1ll * dp[i][k] % mod + left * 1ll * (mod - dp[k + 1][j]) % mod) % mod);
                else add(dp[i][j], (dp[i][k] * 1ll * dp[k + 1][j]) % mod * binom(j - i - 1, k - i) % mod);
            }
        }
    }
    printf("%d\n", dp[1][n]);
}
int main() {
    init_binom(N - 1);
    while (~scanf("%d", &n)) {
        rmain();
    }
    return 0;
}