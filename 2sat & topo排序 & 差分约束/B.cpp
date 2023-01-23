#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 500 + 5;

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
int n, A, B;
vector<int> G[N * 2], e[N * 2];
bool mark[N * 2];
int dis[N][2], d2;
int stk[N * 2], top;
bool dfs(int u) {
    if (mark[u ^ 1]) return false;
    if (mark[u]) return true;
    mark[u] = 1;
    stk[++top] = u;
    for (int v : G[u]) {
        if (!dfs(v)) return false;
    }
    return true;
}
bool two_sat(int mid) {
    for (int i = 1; i <= 2 * n + 1; i++) G[i] = e[i];
    for (int i = 1; i <= n; i++) {
        if (dis[i][0] > mid) G[i * 2].push_back(i * 2 + 1);
        if (dis[i][1] > mid) G[i * 2 + 1].push_back(i * 2);
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (i == j) continue;
            if (dis[i][0] + dis[j][0] > mid) G[i * 2].push_back(j * 2 + 1);
            if (dis[i][1] + dis[j][1] > mid) G[i * 2 + 1].push_back(j * 2);
            if (i < j) {
                if (dis[i][0] + d2 + dis[j][1] > mid) {
                    G[i * 2].push_back(j * 2);
                    G[j * 2 + 1].push_back(i * 2 + 1);
                }
                if (dis[i][1] + d2 + dis[j][0] > mid) {
                    G[i * 2 + 1].push_back(j * 2 + 1);
                    G[j * 2].push_back(i * 2);
                }
            }
        }
    }
    fill(mark + 1, mark + 2 * n + 2, 0);
    for (int i = 1; i <= n; i++) {
        if (!mark[i * 2] && !mark[i * 2 + 1]) {
            top = 0;
            if (!dfs(2 * i + 1)) {
                while (top) mark[stk[top--]] = 0;
                if (!dfs(2 * i)) return 0;
            }
        }
    }
    return 1;
}

void rmain() {
    for (int i = 1; i <= 2 * n + 1; i++) e[i].clear();
    int sx[2], sy[2];
    for (int i = 0; i < 2; i++) scanf("%d %d", &sx[i], &sy[i]);
    d2 = abs(sx[1] - sx[0]) + abs(sy[1] - sy[0]);
    for (int i = 1; i <= n; i++) {
        int x, y;
        scanf("%d %d", &x, &y);
        for (int k = 0; k < 2; k++) dis[i][k] = abs(x - sx[k]) + abs(y - sy[k]);
    }
    for (int i = 1; i <= A; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        e[u * 2 + 1].push_back(v * 2);
        e[v * 2 + 1].push_back(u * 2);
        e[u * 2].push_back(v * 2 + 1);
        e[v * 2].push_back(u * 2 + 1);
    }
    for (int i = 1; i <= B; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        e[u * 2].push_back(v * 2);
        e[v * 2].push_back(u * 2);
        e[u * 2 + 1].push_back(v * 2 + 1);
        e[v * 2 + 1].push_back(u * 2 + 1);
    }
    int l = 0, r = 0x3f3f3f3f, ans = -1;
    while (l <= r) {
        int mid = (l + r) >> 1;
        if (two_sat(mid)) r = mid - 1, ans = mid;
        else l = mid + 1;
    }
    printf("%d\n", ans);
}

int main() {
    while (~scanf("%d %d %d", &n, &A, &B)) {
        rmain();
    }
    return 0;
}