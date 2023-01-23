#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 101 * 26;
#define int long long
// int head[N], ecnt;
// struct Edge {
//     int nxt, v;
// } e[N << 1];
// void add_edge(int u, int v) {
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
//         if (v == fa) continue;
//         if (!dfn[v]) {
//             dfs(v, u);
//             low[u] = min(low[u], low[v]);
//             child++;
//             if (low[v] >= dfn[u]) iscut[u] = 1;
//         } else if (dfn[u] > dfn[v] && v != fa) low[u] = min(low[u], dfn[v]);
//         if (fa == 0 && child == 1) iscut[u] = 0;
//     }

// }
// int dfn[N], low[N], dfs_clock, stk[N], top, scc, sccno[N], siz[N];
// bool in_stk[N];
// vector<int> e[N];
// void tarjan(int u) {
//     in_stk[u] = 1;
//     stk[++top] = u;
//     dfn[u] = low[u] = ++dfs_clock;
//     for (int v : e[u]) {
//         if (!dfn[v]) {
//             tarjan(v);
//             low[u] = min(low[u], low[v]);
//         } else if (in_stk[v]) low[u] = min(low[u], dfn[v]);
//     }
//     if (dfn[u] == low[u]) {
//         int x;
//         scc++;
//         while (true) {
//             x = stk[top];
//             top--;
//             sccno[x] = scc;
//             in_stk[x] = 0;
//             siz[scc]++;
//             if (x == u) break;
//         }
//     }
// }
// vector<int> G[N];
// int top[N], dep[N], faz[N], siz[N], son[N];
// //int dfc, rnk[N], dfn[N];
// void dfs1(int u, int fa) {
//     dep[u] = dep[fa] + 1;
//     siz[u] = 1;
//     son[u] = -1;
//     faz[u] = fa;
//     for (int v : G[u]) {
//         if (v == fa) continue;
//         dfs1(v, u);
//         siz[u] += siz[v];
//         if (son[u] == -1 || siz[son[u]] < siz[v]) son[u] = v;
//     }
// }
// void dfs2(int u, int fa, int tp) {
//     //dfn[u] = ++dfc;
//     //rnk[dfc] = u;
//     top[u] = tp;
//     if (son[u] != -1) dfs2(son[u], u, tp);
//     for (int v : G[u]) {
//         if (v == fa || v == son[u]) continue;
//         dfs2(v, u, v);
//     }
// }
// int LCA(int u, int v) {
//     while (top[u] != top[v]) {
//         if (dep[top[u]] > dep[top[v]])
//             u = faz[top[u]];
//         else
//             v = faz[top[v]];
//     }
//     return dep[u] > dep[v] ? v : u;
// }
int n;
int idx[101][26], s, t;
ll val[101 * 26], cost[101 * 26];
bool vis[N];
int head[N], cur[N], ecnt, d[N];
struct Edge {
    int nxt, v;
    ll flow, cap;
}e[125000 * 2 + N * 4];
void add_edge(int u, int v, ll flow, ll cap) {
    e[ecnt] = {head[u], v, flow, cap}; head[u] = ecnt++;
    e[ecnt] = {head[v], u, flow, 0}; head[v] = ecnt++;
}
bool bfs() {
    memset(vis, 0, sizeof vis);
    std::queue<int> q;
    q.push(s);
    vis[s] = 1;
    d[s] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].v;
            if (vis[v] || e[i].flow >= e[i].cap) continue;
            d[v] = d[u] + 1;
            vis[v] = 1;
            q.push(v);
        }
    }
    return vis[t];
}
ll dfs(int u, ll a) {
    if (u == t || !a) return a;
    int b = 0;
    ll flow = 0, f;
    for (int& i = cur[u]; i != -1; i = e[i].nxt) {
        int v = e[i].v;
        if (d[u] + 1 == d[v] && (f = dfs(v, std::min(a, e[i].cap - e[i].flow))) > 0) {
            e[i].flow += f;
            e[i ^ 1].flow -= f;
            flow += f;
            a -= f;
            if (!a) break;
        }
    }
    return flow;
}

void rmain() {
    scanf("%lld", &n);
    memset(head, -1, sizeof head);
    ecnt = 0;
    int tot = 0;
    for (int i = 1; i <= n; i++) {
        int num;
        scanf("%lld", &num);
        for (int j = 1; j <= num; j++) {
            int x, w;
            x = idx[i][j] = ++tot;
            scanf("%lld %lld %lld", &cost[x], &val[x], &w);
            for (; w--; ) {
                int a, b;
                scanf("%lld %lld", &a, &b);
                add_edge(x, idx[a][b], 0, 1e17);
            }
        }
    }
    t = tot + 1;
    ll res = 0;
    for (int i = 1; i <= tot; i++) {
        res += val[i];
        add_edge(s, i, 0, val[i]);
        add_edge(i, t, 0, cost[i]);
    }
    ll mincut = 0;
    while (bfs()) {
        for (int i = s; i <= t; i++) cur[i] = head[i];
        mincut += dfs(s, 1e17);
    }
    printf("%lld\n", res - mincut);
}

signed main() {
    int Case = 0;
    int T;
    scanf("%lld", &T);
    while (T--) {
        printf("Case #%lld: ", ++Case);
        rmain();
    }
    return 0;
}