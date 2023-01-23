#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 1e5 + 5;

// int head[N], ecnt;
// struct Edge {
//     int nxt, v;
// } e[N << 1];
// void add(int u, int v) {
//     e[++ecnt] = (Edge){head[u], v}, head[u] = ecnt;
//     e[++ecnt] = (Edge){head[v], u}, head[v] = ecnt;
// }
// int fpow(int a, int b) {
//     int res = 1;
//     for (; b; b >>= 1, a = a * 1ll * a % mod) if (b & 1) res = res * 1ll * a % mod;
//     return res;
// }
// ll exgcd(ll a, ll b, ll &x, ll &y) {
//     if (b) {
//         ll d = exgcd(b, a % b, y, x);
//         return y -= a / b * x, d;
//     } else return x = 1, y = 0, a;
// }
// int getinv(int v) {
//     //return fpow(v, mod - 2);
//     // ll x, y;
//     // exgcd(v, mod, x, y);
//     // return (x % mod + mod) % mod;
// }
// int fac[N] = {1}, ifac[N] = {1};
// void init_binom(int n) {
//     for (int i = 1; i <= n; i++) fac[i] = fac[i - 1] * 1ll * i % mod;
//     ifac[n] = getinv(fac[n]);
//     for (int i = n; i > 1; i--) ifac[i - 1] = ifac[i] * 1ll * i % mod;
// }
// int binom(int a, int b) {
//     if (b < 0 || a < 0 || b > a) return 0;
//     return fac[a] * 1ll * ifac[b] % mod * ifac[a - b] % mod; 
// }
// int getphi(int x) {
//     int res = 1;
//     for (int i = 2; i * i <= x; i++) {
//         if (x % i == 0) {
//             x /= i;
//             res *= (i - 1);
//             while (x % i == 0) {
//                 x /= i;
//                 res *= i;
//             }
//         }
//     }
//     if (x > 1) res *= (x - 1);
//     return res;
// }
// int prime[N], pcnt;
// bool isp[N];
// int get_prime(int n) {
//     for (int i = 2; i <= n; i++) {
//         if (!isp[i]) prime[++pcnt] = i;
//         for (int j = 1; j <= pcnt && i * prime[j] <= n; j++) {
//             isp[prime[j] * i] = 1;
//             if (i % prime[j] == 0) break;
//         }
//     }
// }
// int bccno[N], bcc_cnt, siz_e[N], siz_p[N], dfs_clock, low[N], dfn[N], top;
// pair<int, int> stk[N];
// void dfs(int u, int fa) {
//     low[u] = dfn[u] = ++dfs_clock;
//     for(int i = head[u]; i; i = e[i].nxt) {
//         int v = e[i].v;
//         if(v == fa) continue;
//         if(!dfn[v]) {
//             stk[++top] = make_pair(u, v);
//             dfs(v, u);
//             low[u] = min(low[u], low[v]);
//             if(low[v] >= pre[u]) {
//                 bcc_cnt++;
//                 while(true) {
//                     int x = stk[top].first, y = stk[top].second;
//                     top--;
//                     siz_e[bcc_cnt]++;
//                     if(bccno[x] != bcc_cnt) {bccno[x] = bcc_cnt; siz_p[bcc_cnt]++;}
//                     if(bccno[y] != bcc_cnt) {bccno[y] = bcc_cnt; siz_p[bcc_cnt]++;}
//                     if(x == u && y == v) break;
//                 }
//             }
//         } else if(dfn[v] < dfn[u]) {stk[++top] = make_pair(u, v); low[u] = min(low[u], dfn[v]);}
//     }
// }
// int dfn[N], low[N], dfs_clock;
// bool iscut[N], vis[N];
// void dfs(int u, int fa) {
//     dfn[u] = low[u] = ++dfs_clock;
//     vis[u] = 1;
//     int child = 0;
//     for (int v : e[u]) {
//             if (v == fa) continue;
//         if (!dfn[v]) {
//             dfs(v, u);
//             low[u] = min(low[u], low[v]);
//             child++;
//             if (low[v] >= dfn[u]) iscut[u] = 1;
//         } else if (dfn[u] > dfn[v] && v != fa) low[u] = min(low[u], dfn[v]);
//         if (fa == 0 && child == 1) iscut[u] = 0;
//     }

// }
int n, m;
int dfn[N], low[N], dfs_clock, stk[N], top, siz[N], scc, sccno[N];
bool in_stk[N];
vector<int> e[N];
void dfs(int u, int fa) {
    in_stk[u] = 1;
    stk[++top] = u;
    dfn[u] = low[u] = ++dfs_clock;
    for (int v : e[u]) {
            if (v == fa) continue;
        if (!dfn[v]) {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
        } else if (in_stk[v]) low[u] = min(low[u], dfn[v]);
    }
    if (dfn[u] == low[u]) {
        int x;
        scc++;
        while (true) {
            x = stk[top];
            top--;
            siz[scc]++;
            sccno[x] = scc;
            in_stk[x] = 0;
            if (x == u) break;
        }
    }
}

void rmain() {
    if (n == 0) return;
    fill(dfn + 1, dfn + n + 1, 0);
    dfs_clock = scc = 0;
    for (int i = 1; i <= n; i++) e[i].clear();
    for (int i = 1; i <= m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        e[u].push_back(v);
    }
    for (int i = 1; i <= n; i++) if (!dfn[i]) dfs(i, 0);
    scc > 1 ? puts("No") : puts("Yes");
}

int main() {
    while (scanf("%d %d", &n, &m) == 2) {
        rmain();
    }
    return 0;
}