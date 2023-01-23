#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int N = 100005;

int head[N], ecnt;
struct Edge {
    int nxt, v;
} e[N << 1];
void add_edge(int u, int v) {
    e[++ecnt] = (Edge){head[u], v}, head[u] = ecnt;
    e[++ecnt] = (Edge){head[v], u}, head[v] = ecnt;
}
inline bool redi(int &ret) {
   char c; int sgn;
   if(c=getchar(),c==EOF) return 0; //EOF
   while(c!='-'&&(c<'0'||c>'9')) c=getchar();
   sgn=(c=='-')?-1:1;
   ret=(c=='-')?0:(c-'0');
   while(c=getchar(),c>='0'&&c<='9') ret=ret*10+(c-'0');
   ret*=sgn;
   return 1;
}
int n;
ll d[N], d2[N];

int top[N], dep[N], faz[N], son[N], siz[N];
int dfc, dfn[N];
void dfs1(int u, int fa) {
    dep[u] = dep[fa] + 1;
    siz[u] = 1;
    son[u] = -1;
    faz[u] = fa;
    for (int i = head[u]; i; i = e[i].nxt) { 
        int v = e[i].v;
        if (v == fa) continue;
        dfs1(v, u);
        siz[u] += siz[v];
        if (son[u] == -1 || siz[son[u]] < siz[v]) son[u] = v;
    }
}
void dfs2(int u, int fa, int tp) {
    dfn[u] = ++dfc;
    top[u] = tp;
    if (son[u] != -1) dfs2(son[u], u, tp);
    for (int i = head[u]; i; i = e[i].nxt) { 
        int v = e[i].v;
        if (v == fa || v == son[u]) continue;
        dfs2(v, u, v);
    }
}
int LCA(int u, int v) {
    while (top[u] != top[v]) {
        if (dep[top[u]] > dep[top[v]])
            u = faz[top[u]];
        else
            v = faz[top[v]];
    }
    return dep[u] > dep[v] ? v : u;
}
struct Ed {
    int u, v;
} E[N];
void rmain() {
    int q;
    dfc = 0;
    memset(head, 0, sizeof head);
    ecnt = 0;
    redi(n);
    redi(q);
    for (int i = 1; i < n; i++) {
        int u, v;
        redi(u);
        redi(v);
        add_edge(u, v);
        E[i] = {u, v};
    }
    memset(d, 0, sizeof d);
    memset(d2, 0, sizeof d2);
    dfs1(1, 1);
    dfs2(1, 1, 1);
    for (int k = 1; k <= q; k++) {
        char op[10];
        int u, v, x;
        scanf("%s", op);
        redi(u);
        redi(v);
        redi(x);
        if (op[3] == '1') {
            while (top[u] != top[v]) {
                if (dep[top[u]] > dep[top[v]]) {
                    d[dfn[top[u]]]+=x;
                    d[dfn[u] + 1]-=x;
                    u = faz[top[u]];
                } else {
                    d[dfn[top[v]]]+=x;
                    d[dfn[v] + 1]-=x;
                    v = faz[top[v]];
                }
            }
            if (dep[u] > dep[v]) {
                    d[dfn[v]]+=x;
                    d[dfn[u] + 1]-=x;
            } else {
                    d[dfn[u]]+=x;
                    d[dfn[v] + 1]-=x;
            }
        } else {
            while (top[u] != top[v]) {
                if (dep[top[u]] > dep[top[v]]) {
                    d2[dfn[top[u]]]+=x;
                    d2[dfn[u] + 1]-=x;
                    u = faz[top[u]];
                } else {
                    d2[dfn[top[v]]]+=x;
                    d2[dfn[v] + 1]-=x;
                    v = faz[top[v]];
                }
            }
            if (dep[u] > dep[v]) {
                    d2[dfn[v]] += x;
                    d2[dfn[u] + 1] -= x;
            } else {
                    d2[dfn[u]]+=x;
                    d2[dfn[v] + 1]-=x;
            }
            int lca = (dep[u] > dep[v]) ? v : u;
            d2[dfn[lca]]-=x, d2[dfn[lca] + 1] += x;
        }
    }
    for (int i = 1; i <= n; i++) d[i] += d[i - 1], d2[i] += d2[i - 1];
    for (int i = 1; i <= n; i++) printf("%lld%c", d[dfn[i]], " \n"[i == n]);
    for (int i = 1; i < n; i++) {
        int u = E[i].u, v = E[i].v;
        if (faz[v] == u) swap(u, v);
        printf("%lld%c", d2[dfn[u]], " \n"[i == n - 1]);   
    }
    if (n == 1) puts("");
}

int main() {
    int T, Case = 0;
    redi(T);
    while (T--) {
        printf("Case #%d:\n", ++Case);
        rmain();
    }
    return 0;
}