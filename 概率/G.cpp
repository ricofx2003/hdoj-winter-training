#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1e9 + 7;
const int N = 500005;

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

int n, q, x[N], a[N], p[N], dp[N], sum[N];

void rmain() {
    // io.read(n);
    // io.read(q);
    scanf("%d %d", &n, &q);
    for (int i = 1; i <= n; i++) {
        int r, s;
        // io.read(r);
        // io.read(s);
        scanf("%d %d %d %d", &r, &s, &x[i], &a[i]);
        p[i] =  r * 1ll * getinv(s) % mod;
        // io.read(x[i]);
        // io.read(a[i]);
    }
    for (int i = 1; i <= n; i++) {
        dp[i] = (1 + mod - p[i]) % mod * 1ll * getinv(p[i]) % mod * ((sum[i - 1] + mod- sum[x[i] - 1]) % mod) % mod;
        ((dp[i] += a[i] * 1ll * getinv(p[i]) % mod) >= mod) && (dp[i] -= mod);
        sum[i] = (sum[i - 1] + dp[i]) % mod;
    }
    while (q--) {
        int l, r;
        //io.read(l), io.read(r);
        //io.write(sum[r - 1] - sum[l - 1]);
        scanf("%d %d", &l, &r);
        printf("%d\n", (sum[r - 1] - sum[l - 1] + mod) % mod);
        //puts("");
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}