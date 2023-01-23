#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 3005;

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
int fac[N], ifac[N], pow2[N], ipow2[N];
void init_binom(int n) {
    fac[0] = ifac[0] = 1;
    pow2[0] = ipow2[0] = 1;
    for (int i = 1; i <= n; i++) fac[i] = fac[i - 1] * 1ll * i % mod;
    ifac[n] = getinv(fac[n]);
    for (int i = n; i > 1; i--) ifac[i - 1] = ifac[i] * 1ll * i % mod;
    for (int i = 1; i <= n; i++) pow2[i] = pow2[i - 1] * 2 % mod;
    ipow2[n] = getinv(pow2[n]);
    for (int i = n; i > 1; i--) ipow2[i - 1] = ipow2[i] * 2 % mod;
}
int binom(int a, int b) {
    if (b < 0 || a < 0 || b > a) return 0;
    return fac[a] * 1ll * ifac[b] % mod * ifac[a - b] % mod; 
}

int n;
int dp[N], table[N];

int main() {
    init_binom(N - 1);
    for (int i = 2; i < N; i++) {
        int re = ipow2[i];
        for (int j = 0; j < i; j++) {
            re = (re + binom(i - 2, j - 2) * 1ll * ipow2[i] % mod * (dp[j] + 1) % mod) % mod;
            re = (re + (binom(i, j) + mod - binom(i - 2, j - 2)) % mod * 1ll * ipow2[i] % mod);
        }
        dp[i] = re * 1ll * getinv((1 + mod - ipow2[i]) % mod) % mod;
    }
    int ans = 0;
    for (int j = 2; j < N; j++) {
        ans = (ans + j * 1ll * (j - 1) % mod * ipow2[1] % mod * binom(j, 2) % mod * fac[j - 2] % mod * dp[j] % mod * ifac[j] % mod) % mod;
        table[j] = ans;
    }
    while (~scanf("%d", &n)) {
        printf("%d\n", table[n] * 1ll * getinv(n) % mod);
    }
    return 0;
}