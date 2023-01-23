#include <bits/stdc++.h>
using namespace std;

const int N = 1e5, M = 5e6, MAX = 1073741824;
int n, m;
unsigned int x, y, z, f[M * 3];

int mn[N << 2], se[N << 2], tag[N << 2];
#define lson (p << 1)
#define rson ((p << 1) | 1)
#define mid ((l + r) >> 1)
void pushup(int p) {
    if (mn[lson] == mn[rson]) {
        mn[p] = mn[lson];
        se[p] = min(se[lson], se[rson]);
    } else if (mn[lson] < mn[rson]) {
        mn[p] = mn[lson];
        se[p] = min(se[lson], mn[rson]);
    } else {
        mn[p] = mn[rson];
        se[p] = min(se[rson], mn[lson]);
    }
}
void add(int p, int v) {
    if (v <= mn[p]) return;
    mn[p] = v;
    tag[p] = v;
}
void pushdown(int p) {
    if (tag[p] != -1) {
        add(lson, tag[p]);
        add(rson, tag[p]);
        tag[p] = -1;
    }
}
void build(int p, int l, int r) {
    tag[p] = -1;
    if (l == r) {
        mn[p] = 0;
        se[p] = MAX;
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    pushup(p);
}
void modify(int p, int l, int r, int L, int R, int v) {
    if (l > R || r < L) return;
    if (mn[p] >= v) return; 
    if (l >= L && r <= R && se[p] >= v) {
        add(p, v);
        return;
    }
    pushdown(p);
    modify(lson, l, mid, L, R, v);
    modify(rson, mid + 1, r, L, R, v);
    pushup(p);
}
int query(int p, int l, int r, int pos) {
    if (l == r) return mn[p];
    pushdown(p);
    if (pos <= mid) return query(lson, l, mid, pos);
    else return query(rson, mid + 1, r, pos);
}

unsigned int RNG61() {
    unsigned int w;
    x = x ^ (x << 11u);
    x = x ^ (x >> 4u);
    x = x ^ (x << 5u);
    x = x ^ (x >> 14u);
    w = x ^ (y ^ z);
    x = y;
    y = z;
    z = w;
    return z;
}
void rmain() {
    cin >> n >> m >> x >> y >> z;
    for (int i = 1; i <= m * 3; i++) f[i] = RNG61();
    build(1, 1, n);
    for (int i = 1; i <= m; i++) {
        int l = min(f[3 * i - 2] % n + 1, f[3*i-1]%n+1);
        int r = max(f[3 * i - 2] % n + 1, f[3*i-1]%n+1);
        int v = f[3 * i] % (1 << 30);
        modify(1, 1, n, l, r, v);
    }
    long long ans = 0;
    for (int i = 1; i <= n; i++) ans ^= i * 1ll * query(1, 1, n, i);
    cout << ans << '\n';
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        rmain();
    }
    return 0;
}