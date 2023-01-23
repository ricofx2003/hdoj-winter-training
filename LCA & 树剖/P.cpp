#include <bits/stdc++.h>
using namespace std;

const int N = 100005;
int n, m, a[N];
int mx[N << 2], cnt[N << 2];

#define lson (x << 1)
#define rson ((x << 1) | 1)
#define mid ((l + r) >> 1)
int query(int x, int l, int r, int lim) {
    if (l == r) return mx[x] > lim;
    if (mx[lson] <= lim) return query(rson, mid + 1, r, lim);
    else return cnt[x] - cnt[lson] + query(lson, l, mid, lim);
}
void build(int x, int l, int r) {
    if (l == r) {
        cnt[x] = 1;
        mx[x] = a[l];
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    cnt[x] = cnt[lson] + query(rson, mid + 1, r, mx[lson]);
    mx[x] = max(mx[lson], mx[rson]);
}
void modify(int x, int l, int r, int pos, int y) {
    if (l == r) {
        cnt[x] = 1;
        mx[x] = y;
        return;
    }
    if (pos <= mid) modify(lson, l, mid, pos, y);
    else modify(rson, mid + 1, r, pos, y);
    mx[x] = max(mx[lson], mx[rson]);
    cnt[x] = cnt[lson] + query(rson, mid + 1, r, mx[lson]);
}

void rmain() {
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    build(1, 1, n);
    while (m--) {
        int x, y;
        scanf("%d %d", &x, &y);
        modify(1, 1, n, x, y);
        printf("%d\n", cnt[1]);
        modify(1, 1, n, x, a[x]);
    }
}

int main() {
    int T;
    scanf("%d", &T);
    while(T--) {
        rmain();
    }
    return 0;
}