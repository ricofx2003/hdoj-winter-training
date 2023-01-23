#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 30000 + 5;

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
int n, m;
int in[N];
vector<int> G[N];
// int dis[N], qcnt[N];
// bool inq[N];
// bool BellmanFord(int start) {
//     queue<int> q;
//     memset(dis, -0x3f, sizeof dis);
//     memset(qcnt, 0, sizeof qcnt);
//     memset(inq, 0, sizeof inq);
//     q.push(start);
//     dis[start] = 0;
//     inq[start] = 1;
//     qcnt[start]++;
//     while (!q.empty()) {
//         int u = q.front();
//         q.pop();
//         inq[u] = 0;
//         for (int i = head[u]; i; i = e[i].nxt) {
//             int v = e[i].v, w = e[i].w;
//             if (dis[v] < dis[u] + w) {
//                 dis[v] = dis[u] + w;
//                 if (!inq[v]) {
//                     q.push(v);
//                     inq[v] = 1;
//                     qcnt[v]++;
//                     if (qcnt[v] > n) return 0;
//                 }
//             }
//         }
//     }
//     return 1;
// }
void rmain() {
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) G[i].clear(), in[i] = 0;
    for (int i = 1; i <= m; i++) {
        int a, b;
        scanf("%d %d", &a, &b);
        swap(a, b);
        G[a].push_back(b);
        in[b]++;
    }
    priority_queue<int> q;
    for (int i = 1; i <= n; i++) if (!in[i]) q.push(i);
    vector<int> topo;
    while (!q.empty()) {
        int u = q.top();
        q.pop();
        topo.push_back(u);
        for (int v : G[u]) {
            in[v]--;
            if (!in[v]) q.push(v);
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        printf("%d%c", topo[i], " \n"[i == 0]);
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