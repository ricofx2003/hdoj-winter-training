#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 1e9 + 7;
const int N = 1e5 + 5;

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
vector<int> G[N];
struct Edge {
    int u, v, w;
} e[N];
int siz[N], faz[N], n;
void dfs(int u, int fa) {
    siz[u] = 1;
    faz[u] = fa;
    for (int v : G[u]) {
        if (v == fa) continue;
        dfs(v, u);
        siz[u] += siz[v];
    }
}
void add(int &a, int b) {
    ((a += b) >= mod) && (a -= mod);
}

void rmain() {
    for (int i = 1; i <= n; i++) G[i].clear();
    for (int i = 1; i < n; i++) {
        scanf("%d %d %d", &e[i].u, &e[i].v, &e[i].w);
        G[e[i].u].push_back(e[i].v);
        G[e[i].v].push_back(e[i].u);
    }
    dfs(1, 1);
    int res = 0;
    for(int i = 1; i < n; i++) {
        int u = e[i].u, v = e[i].v, w = e[i].w;
        if (faz[u] == v) swap(u, v);
        int dw = siz[v], up = n - dw;
        add(res, dw * 2ll * up % mod * fac[n - 2] % mod * (n - 1) % mod * w % mod);
    }
    printf("%d\n", res);
}
int main() {
    init_binom(N - 1);
    while (~scanf("%d", &n)) {
        rmain();
    }
    return 0;
}