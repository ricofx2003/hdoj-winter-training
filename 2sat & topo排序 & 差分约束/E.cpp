#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 168;

int head[N], ecnt;
struct Edge {
    int nxt, v, w;
} e[N << 2];
void add_edge(int u, int v, int w) {
    e[++ecnt] = (Edge){head[u], v, w}, head[u] = ecnt;
}
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
/*
详细解释一下。
为避免负数，时间计数1～24。令：
R[i] i时间需要的人数 （1<=i<=24）
T[i] i时间应聘的人数 （1<=i<=24）
x[i] i时间录用的人数 （0<=i<=24），其中令x[0]=0
再设s[i]=x[0]+x[1]+……+x[i] （0<=i<=24），
由题意，可得如下方程组：
(1) s[i]-s[i-8]>=R[i]        (8<=i<=24)
(2) s[i]-s[16+i]>=R[i]-s[24] (1<=i<=7)
(3) s[i]-s[i-1]>=0           (1<=i<=24)
(4) s[i-1]-s[i]>=-T[i]       (1<=i<=24)

这个差分约束有个特殊的地方，(2)的右边有未知数s[24]。
这时可以通过枚举s[24]=ans来判断是否有可行解。
即(2)变形为(2') s[i]-s[16+i]>=R[i]-ans (1<=i<=7)
再通过SPFA求解(1)(2')(3)(4)。

不过最后有可能出现这种情况：
(1)(2')(3)(4)虽然有解，但求出的s[24]小于代入(2')里的ans！
这时，显然得到的s[]不满足原来的(2)了（请仔细比较(2)与(2')）。
不过虽然得到的解不满足原方程组，但这并不代表(1)(2)(3)(4)在s[24]=ans时没有可行解！
此外，值得注意的是，当得到的s[24]>ans时，虽然s[24]不一定是最优解，但把ans置成s[24]后，确实是可行解。

所以，简单把(2)置换成(2')是有问题的！
为了等价原命题，必须再加上条件：s[24]>=ans
这就是所谓加出来的那条边（5） s[24]－s[0]>=ans

最后说一下，SPFA后判dis[24]==ans其实是没有必要的。
*/
int n, R[25];
int dis[N], qcnt[N], b[N];
bool inq[N];
bool BellmanFord(int start) {
    queue<int> q;
    memset(dis, -0x3f, sizeof dis);
    memset(qcnt, 0, sizeof qcnt);
    memset(inq, 0, sizeof inq);
    q.push(start);
    dis[start] = 0;
    inq[start] = 1;
    qcnt[start]++;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = 0;
        for (int i = head[u]; i; i = e[i].nxt) {
            int v = e[i].v, w = e[i].w;
            if (dis[v] < dis[u] + w) {
                dis[v] = dis[u] + w;
                if (!inq[v]) {
                    q.push(v);
                    inq[v] = 1;
                    qcnt[v]++;
                    if (qcnt[v] > n) return 0;
                }
            }
        }
    }
    return 1;
}

void rmain() {
    /*i - 7, i
    i + (n - (8 - i) + 1)
    n - 7 + i (i < 8)*/
    memset(b, 0, sizeof b);
    for (int i = 1; i <= 24; i++) scanf("%d", &R[i]);
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
        int x;
        scanf("%d", &x);
        x++;
        b[x]++;
    }
    for (int re = 0; re <= n ; re++) {
        memset(head, 0, sizeof head);
        ecnt = 0;
        for (int i = 1; i <= 24; i++) {
            if (i < 8) {
                int l = 24 - 7 + i;
                add_edge(l - 1, i, R[i] - re);
            } else {
                add_edge(i - 8, i, R[i]);
            }
            add_edge(i, i - 1, -b[i]);
            add_edge(i - 1, i, 0);
        }
        add_edge(24, 0, -re);
        bool flag = BellmanFord(0) && (re == dis[24]);
        //printf("%d %d %d\n", re, flag, dis[24]);
        if (flag) {
            printf("%d\n", re);
            return;
        }
    }
    puts("No Solution");
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        rmain();
    }
    return 0;
}